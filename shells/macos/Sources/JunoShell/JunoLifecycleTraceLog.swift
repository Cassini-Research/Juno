import Foundation

/// Append-only JSONL log of engine-lifecycle phase transitions. One file per
/// process under ``~/Library/Logs/Juno/lifecycle/``, capped at ~1 MB. Rotates
/// to keep the most recent five files. Read by ``JunoDiagnosticsView`` and
/// the ``Help → Show Diagnostics`` menu so users can see exactly what
/// happened on the last few launches without rummaging through Console.app.
///
/// Each event is a single JSON object on its own line:
///   {"ts":"2026-05-04T12:34:56.789Z","from":"spawning","to":"socketBound",
///    "dur_ms":1240,"attempt":1,"error":null,"note":null}
///
/// All writes happen on a private serial queue so callers (the lifecycle
/// actor) never block on disk I/O.
final class JunoLifecycleTraceLog {
    static let shared = JunoLifecycleTraceLog()

    struct Event: Codable {
        let ts: String
        let from: String
        let to: String
        let durMs: Int
        let attempt: Int
        let error: String?
        let note: String?

        enum CodingKeys: String, CodingKey {
            case ts, from, to, attempt, error, note
            case durMs = "dur_ms"
        }
    }

    private let queue = DispatchQueue(label: "com.juno.lifecycle.trace", qos: .utility)
    private let dir: URL?
    private let currentFile: URL?
    private let maxFileBytes: Int = 1_000_000
    private let maxFiles: Int = 5
    private let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    init() {
        let fm = FileManager.default
        let logsRoot = fm.urls(for: .libraryDirectory, in: .userDomainMask).first?
            .appendingPathComponent("Logs/Juno/lifecycle", isDirectory: true)
        if let root = logsRoot {
            try? fm.createDirectory(at: root, withIntermediateDirectories: true)
        }
        self.dir = logsRoot
        if let root = logsRoot {
            let stamp = ISO8601DateFormatter().string(from: Date())
                .replacingOccurrences(of: ":", with: "-")
            self.currentFile = root.appendingPathComponent("trace-\(stamp).jsonl", isDirectory: false)
        } else {
            self.currentFile = nil
        }
        queue.async { [weak self] in self?.rotateIfNeeded() }
    }

    func record(from: String, to: String, durMs: Int, attempt: Int, error: String?, note: String? = nil) {
        let event = Event(
            ts: isoFormatter.string(from: Date()),
            from: from,
            to: to,
            durMs: durMs,
            attempt: attempt,
            error: error,
            note: note
        )
        queue.async { [weak self] in
            self?.append(event: event)
        }
    }

    /// Returns the most recent ``maxFiles`` traces, newest first. Each entry
    /// is a tuple of (file URL, parsed events). Best-effort: malformed lines
    /// are silently skipped so a partial write never blocks the diagnostics
    /// view from rendering older traces.
    func recentTraces() -> [(URL, [Event])] {
        guard let dir else { return [] }
        let fm = FileManager.default
        let urls = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey]))
            ?? []
        let sorted = urls
            .filter { $0.pathExtension == "jsonl" }
            .sorted { lhs, rhs in
                let lm = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let rm = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return lm > rm
            }
        return sorted.prefix(maxFiles).map { url in
            let events = parseFile(at: url)
            return (url, events)
        }
    }

    private func parseFile(at url: URL) -> [Event] {
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        var out: [Event] = []
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            if let bytes = line.data(using: .utf8),
               let ev = try? decoder.decode(Event.self, from: bytes) {
                out.append(ev)
            }
        }
        return out
    }

    private func append(event: Event) {
        guard let file = currentFile else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = []
        guard let bytes = try? encoder.encode(event) else { return }
        var line = bytes
        line.append(0x0A) // newline

        let fm = FileManager.default
        if !fm.fileExists(atPath: file.path) {
            fm.createFile(atPath: file.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: file) else { return }
        defer { try? handle.close() }
        do {
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
        } catch {
            // best-effort: dropped trace lines are not fatal
        }
        rotateIfNeeded()
    }

    private func rotateIfNeeded() {
        guard let dir else { return }
        let fm = FileManager.default
        // Cap the current file: if it grew past the byte limit, the next
        // launch will start a fresh file (filename includes the launch
        // timestamp), so we only need to prune older files here.
        let urls = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey]))
            ?? []
        let traces = urls
            .filter { $0.pathExtension == "jsonl" }
            .sorted { lhs, rhs in
                let lm = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let rm = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return lm > rm
            }
        if traces.count > maxFiles {
            for url in traces.suffix(from: maxFiles) {
                try? fm.removeItem(at: url)
            }
        }
        // If our current file blew past the cap mid-run, truncate to the last
        // half so the file we keep writing to stays bounded.
        if let file = currentFile,
           let attrs = try? fm.attributesOfItem(atPath: file.path),
           let size = attrs[.size] as? Int,
           size > maxFileBytes {
            if let data = try? Data(contentsOf: file) {
                let keep = data.suffix(maxFileBytes / 2)
                try? keep.write(to: file, options: .atomic)
            }
        }
    }
}
