import Darwin
import Foundation

struct JunoEngineClient {
    struct RPCError: Error, LocalizedError {
        let message: String
        let code: Int?

        var errorDescription: String? { message }
    }

    static func call(
        socketPath: String,
        method: String,
        params: [String: Any] = [:],
        binary: Data? = nil,
        timeoutSeconds: TimeInterval = 30
    ) async throws -> (result: [String: Any], binary: Data?) {
        try await withThrowingTaskGroup(of: (result: [String: Any], binary: Data?).self) { group in
            group.addTask {
                try await Task.detached(priority: .userInitiated) {
                    let fd = try openSocket(path: socketPath, timeoutSeconds: timeoutSeconds)
                    defer { close(fd) }

                    let requestId = Int(Date().timeIntervalSince1970 * 1000) ^ Int.random(in: 1...Int.max)
                    let request: [String: Any] = [
                        "jsonrpc": "2.0",
                        "id": requestId,
                        "method": method,
                        "params": params,
                    ]
                    try writeFrame(fd: fd, object: request, binary: binary)
                    while true {
                        try Task.checkCancellation()
                        let (obj, responseBinary) = try readFrame(fd: fd)
                        if obj["event"] != nil { continue }
                        guard (obj["id"] as? NSNumber)?.intValue == requestId || obj["id"] as? Int == requestId else {
                            continue
                        }
                        if let error = obj["error"] as? [String: Any] {
                            let code = (error["code"] as? NSNumber)?.intValue ?? error["code"] as? Int
                            let message = (error["message"] as? String) ?? "engine RPC failed"
                            throw RPCError(message: message, code: code)
                        }
                        return ((obj["result"] as? [String: Any]) ?? [:], responseBinary)
                    }
                }.value
            }

            if timeoutSeconds > 0 {
                group.addTask {
                    try await Task.sleep(nanoseconds: timeoutNanoseconds(timeoutSeconds))
                    throw RPCError(message: "engine RPC timed out after \(String(format: "%.2f", timeoutSeconds))s", code: nil)
                }
            }

            guard let out = try await group.next() else {
                throw RPCError(message: "engine RPC failed before completion", code: nil)
            }
            group.cancelAll()
            return out
        }
    }

    private static func openSocket(path: String, timeoutSeconds: TimeInterval) throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        do {
            try setSocketTimeouts(fd: fd, timeoutSeconds: timeoutSeconds)
        } catch {
            close(fd)
            throw error
        }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let utf8Path = Array(path.utf8CString)
        let capacity = MemoryLayout.size(ofValue: addr.sun_path)
        guard utf8Path.count <= capacity else {
            close(fd)
            throw RPCError(message: "engine socket path is too long", code: nil)
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: capacity) { raw in
                for i in 0..<utf8Path.count {
                    raw[i] = utf8Path[i]
                }
            }
        }
        let len = socklen_t(MemoryLayout<sa_family_t>.size + utf8Path.count)
        let rc = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.connect(fd, sa, len)
            }
        }
        guard rc == 0 else {
            let e = errno
            close(fd)
            throw POSIXError(POSIXErrorCode(rawValue: e) ?? .EIO)
        }
        return fd
    }

    private static func setSocketTimeouts(fd: Int32, timeoutSeconds: TimeInterval) throws {
        guard timeoutSeconds > 0 else { return }
        let wholeSeconds = max(0, Int(timeoutSeconds.rounded(.down)))
        let fractional = max(0, timeoutSeconds - TimeInterval(wholeSeconds))
        var timeout = timeval(
            tv_sec: wholeSeconds,
            tv_usec: Int32(min(999_999, max(1, Int(fractional * 1_000_000))))
        )
        let timeoutSize = socklen_t(MemoryLayout<timeval>.size)
        guard setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, timeoutSize) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        guard setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, timeoutSize) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }

    private static func timeoutNanoseconds(_ seconds: TimeInterval) -> UInt64 {
        UInt64(max(0.001, seconds) * 1_000_000_000)
    }

    private static func readFrame(fd: Int32) throws -> ([String: Any], Data?) {
        let jsonLen = try readUInt32(fd: fd)
        guard jsonLen > 0, jsonLen <= 16 * 1024 * 1024 else {
            throw RPCError(message: "invalid engine frame length", code: nil)
        }
        let jsonData = try readExactly(fd: fd, count: Int(jsonLen))
        guard let obj = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw RPCError(message: "invalid engine JSON frame", code: nil)
        }
        var binary: Data?
        if let n = (obj["binary_size"] as? NSNumber)?.intValue ?? obj["binary_size"] as? Int, n > 0 {
            let declared = try readUInt32(fd: fd)
            guard declared == UInt32(n) else {
                throw RPCError(message: "engine binary frame length mismatch", code: nil)
            }
            binary = try readExactly(fd: fd, count: n)
        }
        return (obj, binary)
    }

    private static func writeFrame(fd: Int32, object: [String: Any], binary: Data?) throws {
        var payload = object
        if let binary, !binary.isEmpty {
            payload["binary_size"] = binary.count
        }
        let data = try JSONSerialization.data(withJSONObject: payload, options: [])
        try writeUInt32(fd: fd, UInt32(data.count))
        try writeAll(fd: fd, data: data)
        if let binary, !binary.isEmpty {
            try writeUInt32(fd: fd, UInt32(binary.count))
            try writeAll(fd: fd, data: binary)
        }
    }

    private static func readUInt32(fd: Int32) throws -> UInt32 {
        let data = try readExactly(fd: fd, count: 4)
        return data.withUnsafeBytes { raw in
            let b0 = UInt32(raw[0]) << 24
            let b1 = UInt32(raw[1]) << 16
            let b2 = UInt32(raw[2]) << 8
            let b3 = UInt32(raw[3])
            return b0 | b1 | b2 | b3
        }
    }

    private static func writeUInt32(fd: Int32, _ value: UInt32) throws {
        var be = value.bigEndian
        let data = Data(bytes: &be, count: 4)
        try writeAll(fd: fd, data: data)
    }

    private static func readExactly(fd: Int32, count: Int) throws -> Data {
        var data = Data(count: count)
        var offset = 0
        try data.withUnsafeMutableBytes { rawBuffer in
            guard let base = rawBuffer.baseAddress else { return }
            while offset < count {
                let n = Darwin.read(fd, base.advanced(by: offset), count - offset)
                if n == 0 {
                    throw RPCError(message: "engine socket closed", code: nil)
                }
                if n < 0 {
                    if errno == EINTR { continue }
                    if errno == EAGAIN || errno == EWOULDBLOCK {
                        throw RPCError(message: "engine socket read timed out", code: nil)
                    }
                    throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                }
                offset += n
            }
        }
        return data
    }

    private static func writeAll(fd: Int32, data: Data) throws {
        try data.withUnsafeBytes { rawBuffer in
            guard let base = rawBuffer.baseAddress else { return }
            var offset = 0
            while offset < data.count {
                let n = Darwin.write(fd, base.advanced(by: offset), data.count - offset)
                if n < 0 {
                    if errno == EINTR { continue }
                    if errno == EAGAIN || errno == EWOULDBLOCK {
                        throw RPCError(message: "engine socket write timed out", code: nil)
                    }
                    throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                }
                if n == 0 {
                    throw RPCError(message: "engine socket write made no progress", code: nil)
                }
                offset += n
            }
        }
    }
}
