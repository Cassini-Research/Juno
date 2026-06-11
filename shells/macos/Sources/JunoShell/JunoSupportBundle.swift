import AppKit
import Darwin
import Foundation

/// Aggregates the most relevant local logs and runtime artifacts into a single
/// `.txt` under `~/Library/Logs/Juno/` and reveals it in Finder, so a user can
/// email it to support without hunting across paths or copying console output.
///
/// Sources combined (best-effort — missing files are noted, not fatal):
/// - `~/Library/Logs/Juno/bundled-engine.log` — written by `run_engine.sh`.
/// - `~/Library/Logs/Juno/juno-bootstrap-engine.log` — captured by
///   `JunoLocalBrokerBootstrap` (`JunoShellApp.swift`).
/// - `~/Library/Logs/Juno/juno-app.log` — only present when the
///   "Save app logs to file" toggle is on.
/// - The newest `~/Library/Application Support/Juno/Workbench/*.jsonl` trace.
/// - `.juno_v2_runtime/{health.json, summary.json, startup_profile.json}`
///   plus the three most recent incidents (probed in both the bundled-engine
///   path and the user's app-support path).
///
/// Header: app version + OS + hardware + sanitized UserDefaults snapshot
/// (preferred display name and other PII keys are deliberately excluded).
enum JunoSupportBundle {
    /// `~/Library/Logs/Juno/` — same parent as `JunoLocalBrokerBootstrap.bootstrapLogURL`.
    static var logDirectory: URL {
        let lib = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library", isDirectory: true)
        return lib.appendingPathComponent("Logs/Juno", isDirectory: true)
    }

    /// Stable label for UI display. We keep the tilde so users recognize the path.
    static let logDirectoryDisplayPath: String = "~/Library/Logs/Juno/"

    /// Where the optional Swift stderr capture lands when "Save app logs to file" is on.
    static var appLogURL: URL {
        return logDirectory.appendingPathComponent("juno-app.log", isDirectory: false)
    }

    /// Open `~/Library/Logs/Juno/` in Finder.
    static func revealLogDirectory() {
        ensureDirectoryExists()
        NSWorkspace.shared.open(logDirectory)
    }

    /// Build a fresh support bundle, reveal it in Finder, and return its URL.
    /// Returns `nil` if the bundle could not be written (typically a permissions
    /// issue on `~/Library/Logs/Juno/`).
    @discardableResult
    static func generateAndReveal() -> URL? {
        ensureDirectoryExists()
        let stamp = filenameTimestamp()
        let bundleURL = logDirectory.appendingPathComponent("juno-support-bundle-\(stamp).txt", isDirectory: false)

        guard FileManager.default.createFile(atPath: bundleURL.path, contents: nil),
              let stream = try? FileHandle(forWritingTo: bundleURL) else {
            return nil
        }
        defer {
            try? stream.synchronize()
            try? stream.close()
        }

        writeHeader(to: stream)
        writeUserDefaultsSection(to: stream)
        writeLogTail(label: "bundled-engine.log",
                     url: logDirectory.appendingPathComponent("bundled-engine.log"),
                     to: stream)
        writeLogTail(label: "juno-bootstrap-engine.log",
                     url: logDirectory.appendingPathComponent("juno-bootstrap-engine.log"),
                     to: stream)
        writeLogTail(label: "juno-app.log", url: appLogURL, to: stream)
        writeLatestWorkbenchTrace(to: stream)
        writeRuntimeArtifacts(to: stream)

        NSWorkspace.shared.activateFileViewerSelecting([bundleURL])
        return bundleURL
    }

    // MARK: - private

    private static func ensureDirectoryExists() {
        try? FileManager.default.createDirectory(at: logDirectory, withIntermediateDirectories: true)
    }

    private static func filenameTimestamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd-HHmmss"
        f.timeZone = TimeZone.current
        f.locale = Locale(identifier: "en_US_POSIX")
        return f.string(from: Date())
    }

    private static func writeLine(_ line: String, to handle: FileHandle) {
        if let data = (line + "\n").data(using: .utf8) {
            handle.write(data)
        }
    }

    private static func writeHeader(to handle: FileHandle) {
        let info = Bundle.main.infoDictionary ?? [:]
        let bundleId = info["CFBundleIdentifier"] as? String ?? "?"
        let osVer = ProcessInfo.processInfo.operatingSystemVersionString
        let hw = sysctlString("hw.model") ?? "?"
        let arch = sysctlString("hw.machine") ?? "?"
        let now = ISO8601DateFormatter().string(from: Date())

        writeLine("=== Juno support bundle ===", to: handle)
        writeLine("Generated: \(now)", to: handle)
        writeLine("App: \(bundleId) \(JunoProductIdentity.versionDetail)", to: handle)
        writeLine("OS: \(osVer)  Hardware: \(hw) (\(arch))", to: handle)
        writeLine("Locale: \(Locale.current.identifier)", to: handle)
        writeLine("", to: handle)
    }

    private static func writeUserDefaultsSection(to handle: FileHandle) {
        writeLine("--- Juno settings (sanitized) ---", to: handle)
        let keys: [String] = [
            JunoUserDefaults.developerModeEnabledKey,
            JunoUserDefaults.saveLogsToFileEnabledKey,
            JunoUserDefaults.micVoiceProcessingEnabledKey,
            JunoUserDefaults.languageModeKey,
            JunoUserDefaults.hudPositionKey,
            JunoUserDefaults.hudLiveTranscriptionsEnabledKey,
            JunoUserDefaults.pauseSensitivitySecondsKey,
            JunoUserDefaults.showInDockKey,
            JunoUserDefaults.appearancePreferenceKey,
            JunoUserDefaults.hudDelightAnimationsEnabledKey,
            JunoUserDefaults.hudDelightSoundEnabledKey,
            JunoUserDefaults.screenContextEnabledKey,
            JunoUserDefaults.onboardingCompletedKey,
        ]
        let ud = UserDefaults.standard
        for key in keys {
            let value = ud.object(forKey: key)
            writeLine("\(key) = \(value.map { "\($0)" } ?? "<unset>")", to: handle)
        }
        writeLine("", to: handle)
    }

    private static func writeLogTail(label: String, url: URL, to handle: FileHandle, lines: Int = 500) {
        writeLine("--- \(label) (tail \(lines) lines from \(url.path)) ---", to: handle)
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8) else {
            writeLine("(file not found or unreadable)", to: handle)
            writeLine("", to: handle)
            return
        }

        // File mtime as a coarse per-line timestamp anchor. NSLog and JSONL
        // already include per-event timestamps in the line content; for
        // untimestamped Python stderr (run_engine.sh / tqdm / bare print),
        // the file's mtime is the most honest signal we have. A leading
        // digit tells us the line probably already starts with its own
        // timestamp — we skip our prefix to avoid stuttering.
        let attrs = (try? FileManager.default.attributesOfItem(atPath: url.path)) ?? [:]
        let mtime = (attrs[.modificationDate] as? Date) ?? Date()
        let lineFormatter = DateFormatter()
        lineFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        lineFormatter.timeZone = TimeZone.current
        lineFormatter.locale = Locale(identifier: "en_US_POSIX")
        let mtimePrefix = "[" + lineFormatter.string(from: mtime) + "] "

        let allLines = text.split(separator: "\n", omittingEmptySubsequences: false)
        for l in allLines.suffix(lines) {
            let trimmed = l.drop(while: { $0 == " " || $0 == "\t" })
            let alreadyStamped = trimmed.first?.isNumber == true
            writeLine(alreadyStamped ? String(l) : mtimePrefix + String(l), to: handle)
        }
        writeLine("", to: handle)
    }

    private static func writeLatestWorkbenchTrace(to handle: FileHandle, lines: Int = 200) {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support", isDirectory: true)
        let workbenchDir = appSupport.appendingPathComponent("Juno/Workbench", isDirectory: true)

        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: workbenchDir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            writeLine("--- workbench trace ---", to: handle)
            writeLine("(workbench dir not found at \(workbenchDir.path))", to: handle)
            writeLine("", to: handle)
            return
        }
        let traces = entries.filter { $0.pathExtension == "jsonl" }.sorted {
            let a = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let b = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return a > b
        }
        guard let latest = traces.first else {
            writeLine("--- workbench trace ---", to: handle)
            writeLine("(no .jsonl files in \(workbenchDir.path))", to: handle)
            writeLine("", to: handle)
            return
        }
        writeLogTail(label: "workbench trace (\(latest.lastPathComponent))", url: latest, to: handle, lines: lines)
    }

    private static func writeRuntimeArtifacts(to handle: FileHandle) {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support", isDirectory: true)
        // Probe order matches what JunoEngineContract uses for the bundled engine,
        // then falls back to user Application Support (where a repo-checkout run
        // would write its .juno_v2_runtime).
        let candidates: [URL] = [
            URL(fileURLWithPath: "/Applications/Juno.app/Contents/Resources/engine/.juno_v2_runtime", isDirectory: true),
            appSupport.appendingPathComponent("Juno/.juno_v2_runtime", isDirectory: true),
        ]
        guard let runtimeDir = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            writeLine("--- runtime artifacts ---", to: handle)
            writeLine("(no .juno_v2_runtime directory found in known locations)", to: handle)
            writeLine("", to: handle)
            return
        }
        writeLine("--- runtime artifacts (\(runtimeDir.path)) ---", to: handle)
        for name in ["health.json", "summary.json", "startup_profile.json"] {
            let f = runtimeDir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: f.path),
                  let data = try? Data(contentsOf: f),
                  let text = String(data: data, encoding: .utf8) else { continue }
            writeLine("- \(name):", to: handle)
            writeLine(text, to: handle)
        }

        let incidentsDir = runtimeDir.appendingPathComponent("incidents", isDirectory: true)
        if let incidents = try? FileManager.default.contentsOfDirectory(
            at: incidentsDir,
            includingPropertiesForKeys: [.contentModificationDateKey]
        ) {
            let recent = incidents.sorted {
                let a = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let b = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return a > b
            }.prefix(3)
            for i in recent {
                guard let data = try? Data(contentsOf: i),
                      let text = String(data: data, encoding: .utf8) else { continue }
                writeLine("- incident \(i.lastPathComponent):", to: handle)
                writeLine(text, to: handle)
            }
        }
        writeLine("", to: handle)
    }

    private static func sysctlString(_ name: String) -> String? {
        var size: size_t = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
        return String(cString: buf)
    }
}
