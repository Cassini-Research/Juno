import Foundation

struct JunoSessionContextTape {
    private(set) var snapshots: [[String: Any]] = []
    private var seenKeys: Set<String> = []
    private let maxSnapshots = 12
    private let maxTextChars = 240

    mutating func reset() {
        snapshots.removeAll(keepingCapacity: true)
        seenKeys.removeAll(keepingCapacity: true)
    }

    mutating func capture(reason: String, base: [String: Any]? = nil) {
        _ = appendSnapshot(reason: reason, base: base ?? JunoLocalCapability.snapshot())
    }

    @discardableResult
    mutating func captureDictationStart(
        reason: String,
        base: [String: Any]? = nil,
        selectionGrab: () -> String? = { JunoLocalCapability.grabSelectedTextViaSyntheticCopy() },
        hasAccessibilityTrust: () -> Bool = { JunoLocalCapability.processHasAccessibilityTrust() }
    ) -> [String: Any] {
        var snap = base ?? JunoLocalCapability.snapshot()
        if JunoLocalCapability.shouldAttemptPasteboardSelectionGrab(
            snapshot: snap,
            hasAccessibilityTrust: hasAccessibilityTrust
        ), let selected = selectionGrab()?.trimmingCharacters(in: .whitespacesAndNewlines),
           !selected.isEmpty {
            snap["selected_text"] = selected
            snap["selection_capture_source"] = "synthetic_cmd_c"
        }
        return appendSnapshot(reason: reason, base: snap)
    }

    static func preservingStartSelectionIfNeeded(
        start: [String: Any]?,
        current: [String: Any]
    ) -> [String: Any] {
        guard let start else { return current }
        let startSelection = ((start["selected_text"] as? String) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let currentSelection = ((current["selected_text"] as? String) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !startSelection.isEmpty, currentSelection.isEmpty else {
            return current
        }
        guard (start["focused_is_secure"] as? Bool) != true,
              (current["focused_is_secure"] as? Bool) != true else {
            return current
        }

        let startBundle = ((start["frontmost_app_bundle_id"] as? String)
            ?? (start["app_bundle_id"] as? String)
            ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let currentBundle = ((current["frontmost_app_bundle_id"] as? String)
            ?? (current["app_bundle_id"] as? String)
            ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard startBundle.isEmpty || currentBundle.isEmpty || startBundle == currentBundle else {
            return current
        }

        var out = current
        out["selected_text"] = startSelection
        if let source = start["selection_capture_source"] as? String, !source.isEmpty {
            out["selection_capture_source"] = source
        }
        return out
    }

    private mutating func appendSnapshot(reason: String, base: [String: Any]) -> [String: Any] {
        var snap = base
        snap["reason"] = reason
        snap["ts_unix_ms"] = Int(Date().timeIntervalSince1970 * 1000)
        snap = Self.sanitizedSnapshot(snap, maxTextChars: maxTextChars)

        let app = (snap["frontmost_app_bundle_id"] as? String)
            ?? (snap["app_bundle_id"] as? String)
            ?? ""
        let title = (snap["window_title"] as? String) ?? ""
        let selected = (snap["selected_text"] as? String) ?? ""
        let before = (snap["focused_text_before"] as? String)
            ?? (snap["focused_text"] as? String)
            ?? ""
        let key = "\(app)|\(title)|\(selected)|\(before)".lowercased()
        guard !seenKeys.contains(key) else { return snap }
        seenKeys.insert(key)

        snapshots.append(snap)
        if snapshots.count > maxSnapshots {
            snapshots.removeFirst(snapshots.count - maxSnapshots)
        }
        return snap
    }

    func payload(liveTranscript: String?) -> [String: Any] {
        var out: [String: Any] = [
            "snapshots": snapshots,
            "snapshot_count": snapshots.count,
        ]
        let draft = (liveTranscript ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !draft.isEmpty {
            out["draft_text"] = String(draft.prefix(360))
        }
        return out
    }

    /// Test-only seam exposing the sanitiser at its production default
    /// length budget so ``SecureFieldPolicyTests`` can assert the
    /// secure-field strip behaviour end-to-end without standing up a
    /// full ``DictationController``.
    static func sanitizedSnapshotForTesting(_ input: [String: Any]) -> [String: Any] {
        sanitizedSnapshot(input, maxTextChars: 240)
    }

    private static func sanitizedSnapshot(_ input: [String: Any], maxTextChars: Int) -> [String: Any] {
        let allowed = [
            "frontmost_app_bundle_id",
            "app_bundle_id",
            "frontmost_app_name",
            "app_name",
            "window_title",
            "selected_text",
            "focused_text",
            "focused_text_before",
            "focused_text_after",
            "field_text_excerpt",
            "clipboard_text",
            "focused_document_path",
            "focused_file_path",
            "candidate_entities",
            "candidate_terms",
            "app_category",
            "can_paste_at_focus",
            "focused_is_secure",
            "has_ax_trust",
            "selection_capture_source",
            "reason",
            "ts_unix_ms",
        ]
        var out: [String: Any] = [:]
        for key in allowed {
            guard let value = input[key] else { continue }
            if let text = value as? String {
                out[key] = String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(maxTextChars))
            } else if JSONSerialization.isValidJSONObject(["v": value]) {
                out[key] = value
            }
        }
        // Gate 1 of 5 (see ``JunoSecureFieldPolicy``): when the focused
        // field is secure, strip every user-text surface from this
        // snapshot. The other four gates (learning, history, audio
        // upload, paste) are enforced at their own callsites in
        // ``DictationController``.
        let secure = (out["focused_is_secure"] as? Bool) == true
        if !SecureFieldPolicy.allowContext(secure: secure) {
            out["selected_text"] = ""
            out["focused_text"] = ""
            out["focused_text_before"] = ""
            out["focused_text_after"] = ""
            out["field_text_excerpt"] = ""
            out["clipboard_text"] = ""
            out["candidate_entities"] = []
            out["candidate_terms"] = []
        }
        return out
    }
}
