import Darwin
import Foundation

enum JunoProcessTreeTerminator {
    struct ProcessRow: Equatable {
        let pid: pid_t
        let ppid: pid_t
    }

    static func terminate(
        process: Process,
        reason: String,
        graceSeconds: TimeInterval = 3.0,
        killGraceSeconds: TimeInterval = 1.0
    ) {
        terminateTree(
            rootPid: process.processIdentifier,
            rootProcess: process,
            reason: reason,
            graceSeconds: graceSeconds,
            killGraceSeconds: killGraceSeconds
        )
    }

    static func terminate(
        rootPid: pid_t,
        reason: String,
        graceSeconds: TimeInterval = 3.0,
        killGraceSeconds: TimeInterval = 1.0
    ) {
        terminateTree(
            rootPid: rootPid,
            rootProcess: nil,
            reason: reason,
            graceSeconds: graceSeconds,
            killGraceSeconds: killGraceSeconds
        )
    }

    private static func terminateTree(
        rootPid: pid_t,
        rootProcess: Process?,
        reason: String,
        graceSeconds: TimeInterval,
        killGraceSeconds: TimeInterval
    ) {
        guard rootPid > 1 else { return }
        let initialDescendants = descendants(of: rootPid)
        let initialTargets = orderedTargets(rootPid: rootPid, descendants: initialDescendants)
        guard initialTargets.contains(where: {
            processExists($0, rootPid: rootPid, rootProcess: rootProcess)
        }) else { return }

        NSLog(
            "Juno: terminating process tree reason=%@ root=%d descendants=%@",
            reason,
            rootPid,
            initialDescendants.map(String.init).joined(separator: ",")
        )
        send(SIGTERM, to: initialTargets)
        waitUntilGone(
            initialTargets,
            rootPid: rootPid,
            rootProcess: rootProcess,
            timeout: graceSeconds
        )

        let remainingDescendants = descendants(of: rootPid)
        let remainingTargets = orderedTargets(rootPid: rootPid, descendants: remainingDescendants)
            .filter { processExists($0, rootPid: rootPid, rootProcess: rootProcess) }
        guard !remainingTargets.isEmpty else { return }

        NSLog(
            "Juno: process tree still running after TERM; sending KILL reason=%@ pids=%@",
            reason,
            remainingTargets.map(String.init).joined(separator: ",")
        )
        send(SIGKILL, to: remainingTargets.reversed())
        waitUntilGone(
            remainingTargets,
            rootPid: rootPid,
            rootProcess: rootProcess,
            timeout: killGraceSeconds
        )
    }

    static func descendants(of rootPid: pid_t) -> [pid_t] {
        descendants(of: rootPid, rows: processRows())
    }

    static func descendants(of rootPid: pid_t, rows: [ProcessRow]) -> [pid_t] {
        var childrenByParent: [pid_t: [pid_t]] = [:]
        for row in rows where row.pid > 1 && row.ppid > 0 {
            childrenByParent[row.ppid, default: []].append(row.pid)
        }

        var out: [pid_t] = []
        var queue = childrenByParent[rootPid] ?? []
        var seen = Set<pid_t>()
        while let pid = queue.first {
            queue.removeFirst()
            guard seen.insert(pid).inserted else { continue }
            out.append(pid)
            queue.append(contentsOf: childrenByParent[pid] ?? [])
        }
        return out
    }

    private static func orderedTargets(rootPid: pid_t, descendants: [pid_t]) -> [pid_t] {
        var seen = Set<pid_t>()
        var targets: [pid_t] = []
        for pid in [rootPid] + descendants where pid > 1 && seen.insert(pid).inserted {
            targets.append(pid)
        }
        return targets
    }

    private static func processRows() -> [ProcessRow] {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/ps")
        task.arguments = ["-axo", "pid=,ppid="]

        let out = Pipe()
        task.standardOutput = out
        task.standardError = Pipe()
        do {
            try task.run()
            task.waitUntilExit()
        } catch {
            NSLog("Juno: failed to enumerate child processes: %@", error.localizedDescription)
            return []
        }
        guard task.terminationStatus == 0 else { return [] }

        let data = out.fileHandleForReading.readDataToEndOfFile()
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { line in
            let parts = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
            guard parts.count >= 2,
                  let pid = Int32(parts[0]),
                  let ppid = Int32(parts[1]) else {
                return nil
            }
            return ProcessRow(pid: pid_t(pid), ppid: pid_t(ppid))
        }
    }

    private static func send<S: Sequence>(_ signal: Int32, to pids: S) where S.Element == pid_t {
        for pid in pids where processExists(pid) {
            _ = Darwin.kill(pid, signal)
        }
    }

    private static func waitUntilGone<S: Sequence>(
        _ pids: S,
        rootPid: pid_t,
        rootProcess: Process?,
        timeout: TimeInterval
    ) where S.Element == pid_t {
        let targets = Array(pids)
        guard !targets.isEmpty, timeout > 0 else { return }
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if !targets.contains(where: {
                processExists($0, rootPid: rootPid, rootProcess: rootProcess)
            }) { return }
            usleep(100_000)
        }
    }

    private static func processExists(
        _ pid: pid_t,
        rootPid: pid_t,
        rootProcess: Process?
    ) -> Bool {
        if pid == rootPid, let rootProcess, !rootProcess.isRunning {
            return false
        }
        return processExists(pid)
    }

    private static func processExists(_ pid: pid_t) -> Bool {
        guard pid > 1 else { return false }
        if Darwin.kill(pid, 0) == 0 { return true }
        return errno == EPERM
    }
}
