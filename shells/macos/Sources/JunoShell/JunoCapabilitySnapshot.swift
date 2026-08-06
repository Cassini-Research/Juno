import Foundation

/// Runs ``juno-capability`` and returns the JSON dictionary (empty on failure).
enum JunoCapabilitySnapshot {
    static func capture() -> [String: Any] {
        guard let bin = HelperBinary.path("juno-capability") else {
            return captureLocal()
        }
        let task = Process()
        task.executableURL = URL(fileURLWithPath: bin)
        let out = Pipe()
        task.standardOutput = out
        task.standardError = Pipe()
        do {
            try task.run()
            task.waitUntilExit()
        } catch {
            return captureLocal()
        }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return captureLocal()
        }
        let local = captureLocal()
        if (obj["has_ax_trust"] as? Bool) == false {
            if (local["has_ax_trust"] as? Bool) == true {
                return local
            }
        }
        return mergeHelperSnapshot(obj, withLocalContext: local)
    }

    /// AppKit-owned state inside ``JunoLocalCapability`` (NSWorkspace,
    /// NSPasteboard, and accessibility-backed window attributes) must be
    /// read on the main thread. Most capability captures already originate
    /// there, but ``SurfaceEditingModel`` intentionally performs its helper
    /// process work on a utility queue. Hop only the in-process merge back
    /// to main so foreground polling cannot trip AppKit's main-thread
    /// precondition while Juno itself is the active app.
    private static func captureLocal() -> [String: Any] {
        if Thread.isMainThread {
            return JunoLocalCapability.snapshot()
        }
        return DispatchQueue.main.sync {
            JunoLocalCapability.snapshot()
        }
    }

    private static func mergeHelperSnapshot(
        _ helper: [String: Any],
        withLocalContext local: [String: Any]
    ) -> [String: Any] {
        guard (local["has_ax_trust"] as? Bool) == true else { return helper }
        let helperBundle = ((helper["frontmost_app_bundle_id"] as? String)
            ?? (helper["app_bundle_id"] as? String)
            ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let localBundle = ((local["frontmost_app_bundle_id"] as? String)
            ?? (local["app_bundle_id"] as? String)
            ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard helperBundle.isEmpty || localBundle.isEmpty || helperBundle == localBundle else {
            return helper
        }

        var merged = helper
        let secure = (helper["focused_is_secure"] as? Bool) == true
            || (local["focused_is_secure"] as? Bool) == true
        if secure {
            merged["focused_is_secure"] = true
            merged["selected_text"] = ""
            merged["focused_text_before"] = ""
            merged["focused_text_after"] = ""
            merged["field_text_excerpt"] = ""
            merged["clipboard_text"] = ""
            merged["candidate_entities"] = []
            merged["candidate_terms"] = []
            return merged
        }

        for key in [
            "selected_text",
            "focused_text_before",
            "focused_text_after",
            "field_text_excerpt",
            "focused_document_path",
            "focused_file_path",
        ] {
            let helperText = ((merged[key] as? String) ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let localText = ((local[key] as? String) ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if helperText.isEmpty, !localText.isEmpty {
                merged[key] = localText
            }
        }

        let helperCandidates = (merged["candidate_entities"] as? [Any]) ?? []
        let localCandidates = (local["candidate_entities"] as? [Any]) ?? []
        if !localCandidates.isEmpty {
            var out: [String] = []
            var seen: Set<String> = []
            for raw in helperCandidates + localCandidates {
                let value = String(describing: raw)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !value.isEmpty else { continue }
                let key = value.lowercased()
                guard !seen.contains(key) else { continue }
                seen.insert(key)
                out.append(value)
                if out.count >= 40 { break }
            }
            merged["candidate_entities"] = out
        }
        return merged
    }
}
