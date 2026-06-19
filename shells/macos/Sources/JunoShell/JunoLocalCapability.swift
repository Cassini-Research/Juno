import ApplicationServices
import AppKit
import Foundation

/// In-process fallback for the short-lived `juno-capability` helper.
///
/// Local development builds can have Accessibility enabled for `Juno.app` while
/// the separately signed helper binaries still lack their own TCC grant. In
/// that state the broker reports `ax_permission_missing`, even though the shell
/// process can safely inspect the focused element and paste. This fallback keeps
/// the safety gate in the AX-trusted process instead of silently disabling
/// auto-paste.
enum JunoLocalCapability {
    private static let maxContextFieldLen = 1600
    private static let maxVisibleContextLen = 1600
    private static let maxVisibleCandidateCount = 24

    static func processHasAccessibilityTrust() -> Bool {
        AXIsProcessTrusted()
    }

    static func shouldAttemptPasteboardSelectionGrab(
        snapshot: [String: Any],
        hasAccessibilityTrust: () -> Bool = { processHasAccessibilityTrust() }
    ) -> Bool {
        let selected = ((snapshot["selected_text"] as? String) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard selected.isEmpty else { return false }
        guard (snapshot["focused_is_secure"] as? Bool) != true else { return false }
        guard (snapshot["has_ax_trust"] as? Bool) == true, hasAccessibilityTrust() else {
            return false
        }
        let bundleId = ((snapshot["frontmost_app_bundle_id"] as? String)
            ?? (snapshot["app_bundle_id"] as? String)
            ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return knownPasteCentricBundleIds.contains(bundleId)
    }

    static func grabSelectedTextViaSyntheticCopy(timeoutMs: Int = 80) -> String? {
        let pasteboard = NSPasteboard.general
        let beforeChangeCount = pasteboard.changeCount
        let savedItems = snapshotPasteboardItems(pasteboard)

        guard let source = CGEventSource(stateID: .privateState),
              let keyDown = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(0x08), keyDown: true),
              let keyUp = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(0x08), keyDown: false) else {
            return nil
        }
        source.localEventsSuppressionInterval = 0
        keyDown.flags = .maskCommand
        keyUp.flags = .maskCommand
        keyDown.post(tap: .cgSessionEventTap)
        keyUp.post(tap: .cgSessionEventTap)

        let deadline = Date().addingTimeInterval(Double(max(0, timeoutMs)) / 1000.0)
        var didAdvance = pasteboard.changeCount != beforeChangeCount
        while !didAdvance && Date() < deadline {
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.005))
            didAdvance = pasteboard.changeCount != beforeChangeCount
        }
        guard didAdvance else {
            return nil
        }

        let copied = pasteboard.string(forType: .string)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        restorePasteboardItems(savedItems, to: pasteboard)
        guard !copied.isEmpty else { return nil }
        return clip(copied)
    }

    /// Read the focused editable element's current text value, for paste
    /// read-back verification (did the text actually land, vs. did we merely
    /// post a Cmd+V keystroke). Resolves the focused element the same way
    /// ``snapshot()`` does — via ``pasteCandidate`` / ``pasteCandidateFromWindow``
    /// — so "before" and "after" reads target the same field.
    ///
    /// Returns:
    ///   - ``nil``: no frontmost app / no AX trust — caller can't verify.
    ///   - ``(readable: false, "")``: a focused target exists but exposes no
    ///     readable ``AXValue`` (Chromium/Electron, web fields, Terminal, custom
    ///     widgets). Callers MUST treat this as "cannot verify → assume the
    ///     paste succeeded": a read-back check on these would report real pastes
    ///     as failures (false negatives), which is worse than the rare false
    ///     positive it would prevent.
    ///   - ``(readable: true, value)``: the field's current text value.
    static func focusedValueSignature() -> (readable: Bool, value: String)? {
        guard processHasAccessibilityTrust(),
              let frontmost = NSWorkspace.shared.frontmostApplication else { return nil }
        let appElement = AXUIElementCreateApplication(frontmost.processIdentifier)
        let focusedWindow = axElement(appElement, kAXFocusedWindowAttribute as CFString)
        let mainWindow = axElement(appElement, kAXMainWindowAttribute as CFString)
        let windowCandidate = pasteCandidateFromWindow(focusedWindow: focusedWindow, mainWindow: mainWindow)
        let focused: AXUIElement
        if let rawFocused = axElement(appElement, kAXFocusedUIElementAttribute as CFString) {
            focused = pasteCandidate(from: rawFocused)?.element ?? windowCandidate?.element ?? rawFocused
        } else if let wc = windowCandidate {
            focused = wc.element
        } else {
            return (false, "")
        }
        let focusedRole = axString(focused, kAXRoleAttribute as CFString) ?? ""
        guard pasteReadbackReliableRoles.contains(focusedRole) else {
            return (false, "")
        }
        if let v = axString(focused, kAXValueAttribute as CFString) {
            return (true, v)
        }
        return (false, "")
    }

    static func snapshot() -> [String: Any] {
        var out: [String: Any] = [
            "ok": true,
            "has_ax_trust": processHasAccessibilityTrust(),
            "focused_is_secure": false,
            "selected_text": "",
            "focused_text_before": "",
            "focused_text_after": "",
            "clipboard_text": "",
            "locale_identifier": Locale.current.identifier,
        ]

        guard let frontmost = NSWorkspace.shared.frontmostApplication else {
            out["ok"] = false
            out["error"] = "no_frontmost_app"
            return out
        }
        if let bundleId = frontmost.bundleIdentifier {
            out["frontmost_app_bundle_id"] = bundleId
            out["app_bundle_id"] = bundleId
        }
        if let name = frontmost.localizedName {
            out["frontmost_app_name"] = name
            out["app_name"] = name
        }
        out["frontmost_pid"] = Int(frontmost.processIdentifier)

        guard (out["has_ax_trust"] as? Bool) == true else {
            out["ok"] = false
            out["error"] = "ax_permission_not_granted"
            return out
        }

        let appElement = AXUIElementCreateApplication(frontmost.processIdentifier)
        let focusedWindow = axElement(appElement, kAXFocusedWindowAttribute as CFString)
        let mainWindow = axElement(appElement, kAXMainWindowAttribute as CFString)
        if let focusedWindow {
            out["window_title"] = axString(focusedWindow, kAXTitleAttribute as CFString)
            if let doc = axString(focusedWindow, kAXDocumentAttribute as CFString), !doc.isEmpty {
                out["focused_document_path"] = clip(doc)
            }
        }
        if out["window_title"] == nil,
           let mainWindow {
            out["window_title"] = axString(mainWindow, kAXTitleAttribute as CFString)
            if out["focused_document_path"] == nil,
               let doc = axString(mainWindow, kAXDocumentAttribute as CFString), !doc.isEmpty {
                out["focused_document_path"] = clip(doc)
            }
        }

        let windowCandidate = pasteCandidateFromWindow(focusedWindow: focusedWindow, mainWindow: mainWindow)
        if let rawFocused = axElement(appElement, kAXFocusedUIElementAttribute as CFString) {
            let rawRole = axString(rawFocused, kAXRoleAttribute as CFString)
            let rawSubrole = axString(rawFocused, kAXSubroleAttribute as CFString)
            let focusedCandidate = pasteCandidate(from: rawFocused)
            let candidate = focusedCandidate ?? windowCandidate
            let focused = candidate?.element ?? rawFocused
            let role = axString(focused, kAXRoleAttribute as CFString)
            let subrole = axString(focused, kAXSubroleAttribute as CFString)
            if let rawRole, rawRole != role {
                out["focused_container_role"] = rawRole
            }
            if let role { out["focused_role"] = role }
            if let subrole { out["focused_subrole"] = subrole }
            let canPaste = candidate != nil || knownPasteCentricAppAllowsFallback(bundleId: frontmost.bundleIdentifier, role: rawRole)
            out["can_paste_at_focus"] = canPaste
            if focusedCandidate == nil && windowCandidate != nil {
                out["paste_candidate_source"] = "focused_window"
            }
            let secure = rawSubrole == "AXSecureTextField" || subrole == "AXSecureTextField"
            out["focused_is_secure"] = secure
            if !secure {
                if let sel = axString(focused, kAXSelectedTextAttribute as CFString), !sel.isEmpty {
                    out["selected_text"] = clip(sel)
                }
                addCaretContext(from: focused, into: &out)
            }
        } else if let candidate = windowCandidate {
            let focused = candidate.element
            let role = axString(focused, kAXRoleAttribute as CFString)
            let subrole = axString(focused, kAXSubroleAttribute as CFString)
            if let role { out["focused_role"] = role }
            if let subrole { out["focused_subrole"] = subrole }
            out["can_paste_at_focus"] = true
            out["paste_candidate_source"] = "focused_window"
            let secure = subrole == "AXSecureTextField"
            out["focused_is_secure"] = secure
            if !secure {
                if let sel = axString(focused, kAXSelectedTextAttribute as CFString), !sel.isEmpty {
                    out["selected_text"] = clip(sel)
                }
                addCaretContext(from: focused, into: &out)
            }
        } else if knownPasteCentricBundleIds.contains((frontmost.bundleIdentifier ?? "").lowercased()) {
            // Web-rendered apps frequently
            // do not expose any AXFocusedUIElement to outside processes —
            // ``AXUIElementCopyAttributeValue(.., kAXFocusedUIElementAttribute, ..)``
            // returns kAXErrorNoValue. Without this fallback the snapshot
            // leaves ``can_paste_at_focus`` unset, the broker decision goes
            // ``no_text_focus``, the shell flips ``likelyPasteDestination``
            // to false, and the final paste is short-circuited into a copy
            // toast even though Cmd+V into these apps works fine. Treat
            // these apps as paste-friendly when we are AX-trusted but the
            // app refuses to expose its focus tree.
            out["can_paste_at_focus"] = true
        }

        if (out["focused_is_secure"] as? Bool) != true,
           let visibleContext = visibleWindowContext(focusedWindow: focusedWindow, mainWindow: mainWindow) {
            out["field_text_excerpt"] = visibleContext.text
            if !visibleContext.candidates.isEmpty {
                out["candidate_entities"] = visibleContext.candidates
            }
        }

        // OCR-harvested screen vocabulary is optional. It is included only
        // after the user has opted in and macOS access is already granted.
        if (out["focused_is_secure"] as? Bool) != true,
           JunoScreenContextAccess.isEnabledAndGranted {
            let screenTerms = JunoScreenTermHarvester.shared.currentTerms()
            if !screenTerms.isEmpty {
                var candidates = (out["candidate_entities"] as? [String]) ?? []
                var seen = Set(candidates.map { $0.lowercased() })
                for term in screenTerms where !seen.contains(term.lowercased()) {
                    candidates.append(term)
                    seen.insert(term.lowercased())
                }
                out["candidate_entities"] = Array(candidates.prefix(48))
                out["recent_screen_terms"] = Array(screenTerms.prefix(40))
            }
        }

        if (out["focused_is_secure"] as? Bool) != true,
           let clipText = NSPasteboard.general.string(forType: .string), !clipText.isEmpty {
            out["clipboard_text"] = clip(clipText)
        }
        return out
    }

    static func brokerDecisionObject(from snapshot: [String: Any]) -> [String: Any] {
        let report = reportObject(from: snapshot)
        let hasTrust = (snapshot["has_ax_trust"] as? Bool) ?? false
        if !hasTrust {
            return decision(
                ok: false,
                reason: "ax_permission_missing",
                message: "Accessibility must be enabled for the current Juno build. Remove stale Juno entries in System Settings, add Juno again, then relaunch.",
                report: report
            )
        }

        if (snapshot["focused_is_secure"] as? Bool) == true {
            return decision(
                ok: false,
                reason: "secure_field",
                message: "The focused field is a secure-text input. Dictation is blocked for safety.",
                report: report
            )
        }

        let bundleId = ((snapshot["frontmost_app_bundle_id"] as? String)
            ?? (snapshot["app_bundle_id"] as? String)
            ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if defaultBlockedBundleIds.contains(bundleId) {
            return decision(
                ok: false,
                reason: "app_blocked",
                message: "This app is on Juno's managed-app blocklist. Dictation is disabled.",
                report: report
            )
        }

        let role = (snapshot["focused_role"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let canPasteAtFocus = (snapshot["can_paste_at_focus"] as? Bool)
            ?? pasteFriendlyRoles.contains(role)
        if !canPasteAtFocus {
            return decision(
                ok: false,
                reason: "no_text_focus",
                message: "No editable text field appears focused. Dictation will continue; text will be offered to copy instead of pasting.",
                report: report
            )
        }

        return decision(ok: true, reason: "allowed", message: "ok", report: report)
    }

    private static let pasteFriendlyRoles: Set<String> = [
        "AXTextField",
        "AXTextArea",
        "AXComboBox",
        "AXSearchField",
        "AXWebArea",
    ]

    private static let pasteReadbackReliableRoles: Set<String> = [
        "AXTextField",
        "AXTextArea",
        "AXComboBox",
        "AXSearchField",
    ]

    private struct PasteCandidate {
        let element: AXUIElement
    }

    private static func pasteCandidate(from root: AXUIElement) -> PasteCandidate? {
        var seeds: [AXUIElement] = [root]
        var parent = axElement(root, kAXParentAttribute as CFString)
        var parentDepth = 0
        while let p = parent, parentDepth < 3 {
            seeds.append(p)
            parent = axElement(p, kAXParentAttribute as CFString)
            parentDepth += 1
        }

        for seed in seeds {
            if isTextInsertionCandidate(seed) {
                return PasteCandidate(element: seed)
            }
            if let descendant = descendantTextCandidate(from: seed) {
                return descendant
            }
        }
        return nil
    }

    private static func pasteCandidateFromWindow(
        focusedWindow: AXUIElement?,
        mainWindow: AXUIElement?
    ) -> PasteCandidate? {
        for window in [focusedWindow, mainWindow].compactMap({ $0 }) {
            if let candidate = descendantTextCandidate(from: window, maxDepth: 8, maxVisited: 500) {
                return candidate
            }
        }
        return nil
    }

    private static func descendantTextCandidate(
        from root: AXUIElement,
        maxDepth: Int = 4,
        maxVisited: Int = 80
    ) -> PasteCandidate? {
        var queue: [(AXUIElement, Int)] = [(root, 0)]
        var visited = 0

        while !queue.isEmpty && visited < maxVisited {
            let (element, depth) = queue.removeFirst()
            visited += 1
            if depth > 0,
               isTextInsertionCandidate(element) {
                return PasteCandidate(element: element)
            }
            guard depth < maxDepth else { continue }
            for child in axElements(element, kAXChildrenAttribute as CFString) {
                queue.append((child, depth + 1))
            }
        }
        return nil
    }

    private static func isTextInsertionCandidate(_ element: AXUIElement) -> Bool {
        if let role = axString(element, kAXRoleAttribute as CFString),
           pasteFriendlyRoles.contains(role) {
            return true
        }
        return supportsSelectedTextRange(element)
    }

    private static func supportsSelectedTextRange(_ element: AXUIElement) -> Bool {
        var value: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            &value
        )
        guard err == .success, let v = value else { return false }
        return CFGetTypeID(v) == AXValueGetTypeID()
    }

    private static func knownPasteCentricAppAllowsFallback(bundleId: String?, role: String?) -> Bool {
        guard let bundleId = bundleId?.lowercased(), knownPasteCentricBundleIds.contains(bundleId) else {
            return false
        }
        let r = role?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        // Notes can report the focused note editor as a container instead of
        // AXTextArea. Still require an app UI element, not only the top window.
        return !r.isEmpty && r != "AXWindow" && r != "AXApplication"
    }

    private static let knownPasteCentricBundleIds: Set<String> = [
        "com.apple.notes",
        "com.apple.textedit",
    ]

    private struct PasteboardItemSnapshot {
        let values: [(NSPasteboard.PasteboardType, Data)]
    }

    private static func snapshotPasteboardItems(_ pasteboard: NSPasteboard) -> [PasteboardItemSnapshot] {
        (pasteboard.pasteboardItems ?? []).map { item in
            let values = item.types.compactMap { type -> (NSPasteboard.PasteboardType, Data)? in
                guard let data = item.data(forType: type) else { return nil }
                return (type, data)
            }
            return PasteboardItemSnapshot(values: values)
        }
    }

    private static func restorePasteboardItems(
        _ snapshots: [PasteboardItemSnapshot],
        to pasteboard: NSPasteboard
    ) {
        pasteboard.clearContents()
        let items = snapshots.map { snapshot in
            let item = NSPasteboardItem()
            for (type, data) in snapshot.values {
                item.setData(data, forType: type)
            }
            return item
        }
        if !items.isEmpty {
            pasteboard.writeObjects(items)
        }
    }

    private static let defaultBlockedBundleIds: Set<String> = [
        "com.1password.1password",
        "com.1password.1password7",
        "com.agilebits.onepassword7",
        "com.bitwarden.desktop",
        "com.lastpass.LastPass",
        "com.dashlane.dashlanephonefinalmac",
        "com.keepassxc.keepassxc",
        "com.intuit.quickbooks.mac",
        "com.intuit.TurboTax",
        "com.hrblock.desktop",
        "com.1password.1password-cli",
        "com.tailscale.ipn.macos",
    ]

    private static func decision(
        ok: Bool,
        reason: String,
        message: String,
        report: [String: Any]
    ) -> [String: Any] {
        [
            "ok": ok,
            "reason": reason,
            "message": message,
            "report": report,
            "recognition_hints": [],
        ]
    }

    private static func reportObject(from snapshot: [String: Any]) -> [String: Any] {
        var report: [String: Any] = [:]
        for key in [
            "ok",
            "has_ax_trust",
            "frontmost_app_bundle_id",
            "frontmost_app_name",
            "frontmost_pid",
            "window_title",
            "focused_role",
            "focused_container_role",
            "focused_subrole",
            "focused_is_secure",
            "can_paste_at_focus",
            "paste_candidate_source",
            "error",
        ] {
            if let value = snapshot[key] {
                report[key] = value
            }
        }
        return report
    }

    private static func addCaretContext(from element: AXUIElement, into out: inout [String: Any]) {
        var selRange = CFRange(location: 0, length: 0)
        var rangeValue: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            &rangeValue
        )
        guard err == .success,
              let axRangeRef = rangeValue,
              CFGetTypeID(axRangeRef) == AXValueGetTypeID() else {
            return
        }
        // swiftlint:disable:next force_cast
        let axRange = axRangeRef as! AXValue
        guard AXValueGetType(axRange) == .cfRange,
              AXValueGetValue(axRange, .cfRange, &selRange) else {
            return
        }

        let caret = max(0, selRange.location)
        let beforeLen = min(caret, maxContextFieldLen)
        if beforeLen > 0,
           let before = axRangeString(element, location: caret - beforeLen, length: beforeLen),
           !before.isEmpty {
            out["focused_text_before"] = clip(before)
        }
        let afterStart = caret + max(0, selRange.length)
        if let after = axRangeString(element, location: afterStart, length: maxContextFieldLen),
           !after.isEmpty {
            out["focused_text_after"] = clip(after)
        }
    }

    private static func clip(_ s: String?) -> String? {
        guard let s, !s.isEmpty else { return s }
        if s.count <= maxContextFieldLen { return s }
        return String(s.prefix(maxContextFieldLen))
    }

    /// Number of characters in the currently-focused text field of the
    /// frontmost app, when Accessibility can read it. Returns ``nil`` when
    /// AX trust is missing, no app is frontmost, no element has focus, or
    /// the focused element doesn't expose either ``kAXNumberOfCharacters``
    /// or ``kAXValue`` (Electron quirks, custom widgets, Terminal).
    ///
    /// Used by the paste verifier in ``DictationController``: snapshot
    /// before and after the synthetic Cmd+V; a change means the paste
    /// reached the target field. ``nil`` from either side means we have no
    /// signal and fall back to trusting the CGEvent post.
    static func focusedFieldCharacterCount() -> Int? {
        guard AXIsProcessTrusted() else { return nil }
        guard let frontmost = NSWorkspace.shared.frontmostApplication else { return nil }
        let appElement = AXUIElementCreateApplication(frontmost.processIdentifier)
        guard let focused = axElement(appElement, kAXFocusedUIElementAttribute as CFString) else {
            return nil
        }
        if let count = axInt(focused, kAXNumberOfCharactersAttribute as CFString) {
            return count
        }
        if let value = axString(focused, kAXValueAttribute as CFString) {
            return value.count
        }
        return nil
    }

    private static func axInt(_ element: AXUIElement, _ attr: CFString) -> Int? {
        var value: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, attr, &value)
        guard err == .success, let v = value else { return nil }
        if let n = v as? Int { return n }
        if let n = v as? Int64 { return Int(n) }
        if let n = v as? Double { return Int(n) }
        if CFGetTypeID(v) == CFNumberGetTypeID() {
            var out: Int = 0
            if CFNumberGetValue((v as! CFNumber), .longType, &out) {
                return out
            }
        }
        return nil
    }

    private static func axString(_ element: AXUIElement, _ attr: CFString) -> String? {
        var value: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, attr, &value)
        guard err == .success, let v = value as? String else { return nil }
        return v
    }

    private static func axElement(_ element: AXUIElement, _ attr: CFString) -> AXUIElement? {
        var value: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, attr, &value)
        guard err == .success, let v = value, CFGetTypeID(v) == AXUIElementGetTypeID() else {
            return nil
        }
        // swiftlint:disable:next force_cast
        return (v as! AXUIElement)
    }

    private static func axElements(_ element: AXUIElement, _ attr: CFString) -> [AXUIElement] {
        var value: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, attr, &value)
        guard err == .success, let v = value else { return [] }
        if CFGetTypeID(v) == AXUIElementGetTypeID() {
            // swiftlint:disable:next force_cast
            return [v as! AXUIElement]
        }
        guard let arr = v as? [Any] else { return [] }
        return arr.compactMap { item in
            guard let ref = item as CFTypeRef?,
                  CFGetTypeID(ref) == AXUIElementGetTypeID() else {
                return nil
            }
            // swiftlint:disable:next force_cast
            return (ref as! AXUIElement)
        }
    }

    private static func axRangeString(
        _ element: AXUIElement,
        location: Int,
        length: Int
    ) -> String? {
        guard length > 0 else { return "" }
        var range = CFRange(location: location, length: length)
        guard let axRange = AXValueCreate(.cfRange, &range) else { return nil }
        var value: CFTypeRef?
        let err = AXUIElementCopyParameterizedAttributeValue(
            element,
            kAXStringForRangeParameterizedAttribute as CFString,
            axRange,
            &value
        )
        guard err == .success, let s = value as? String else { return nil }
        return s
    }

    private struct VisibleWindowContext {
        let text: String
        let candidates: [String]
    }

    private static func visibleWindowContext(
        focusedWindow: AXUIElement?,
        mainWindow: AXUIElement?
    ) -> VisibleWindowContext? {
        var windows: [AXUIElement] = []
        if let focusedWindow { windows.append(focusedWindow) }
        if let mainWindow { windows.append(mainWindow) }
        guard !windows.isEmpty else { return nil }

        var chunks: [String] = []
        var seenChunks: Set<String> = []
        for window in windows.prefix(2) {
            collectVisibleStrings(from: window, into: &chunks, seen: &seenChunks)
            if joinedVisibleText(chunks).count >= maxVisibleContextLen {
                break
            }
        }
        let text = joinedVisibleText(chunks)
        guard !text.isEmpty else { return nil }
        return VisibleWindowContext(
            text: text,
            candidates: screenCandidateEntities(from: text)
        )
    }

    private static func collectVisibleStrings(
        from root: AXUIElement,
        into chunks: inout [String],
        seen: inout Set<String>,
        maxDepth: Int = 7,
        maxVisited: Int = 700
    ) {
        var queue: [(AXUIElement, Int)] = [(root, 0)]
        var visited = 0
        while !queue.isEmpty && visited < maxVisited && joinedVisibleText(chunks).count < maxVisibleContextLen {
            let (element, depth) = queue.removeFirst()
            visited += 1
            let role = axString(element, kAXRoleAttribute as CFString) ?? ""
            let subrole = axString(element, kAXSubroleAttribute as CFString) ?? ""
            if subrole == "AXSecureTextField" { continue }

            for attr in [
                kAXValueAttribute as CFString,
                kAXTitleAttribute as CFString,
                kAXDescriptionAttribute as CFString,
                "AXPlaceholderValue" as CFString,
            ] {
                guard let value = axString(element, attr) else { continue }
                appendVisibleChunk(value, role: role, into: &chunks, seen: &seen)
            }

            guard depth < maxDepth else { continue }
            for child in axElements(element, kAXChildrenAttribute as CFString) {
                queue.append((child, depth + 1))
            }
        }
    }

    private static func appendVisibleChunk(
        _ value: String,
        role: String,
        into chunks: inout [String],
        seen: inout Set<String>
    ) {
        let collapsed = value
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\t", with: " ")
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard collapsed.count >= 2 else { return }
        guard collapsed.count <= 500 else {
            let clipped = String(collapsed.prefix(500))
            appendVisibleChunk(clipped, role: role, into: &chunks, seen: &seen)
            return
        }
        if role == "AXWindow", chunks.contains(collapsed) {
            return
        }
        let key = collapsed.lowercased()
        guard !seen.contains(key) else { return }
        seen.insert(key)
        chunks.append(collapsed)
    }

    private static func joinedVisibleText(_ chunks: [String]) -> String {
        var out = ""
        for chunk in chunks {
            if out.isEmpty {
                out = chunk
            } else {
                out += " " + chunk
            }
            if out.count >= maxVisibleContextLen {
                return String(out.prefix(maxVisibleContextLen))
            }
        }
        return out
    }

    private static func screenCandidateEntities(from text: String) -> [String] {
        var out: [String] = []
        var seen: Set<String> = []
        var phrase: [String] = []

        func add(_ raw: String) {
            let value = raw.trimmingCharacters(in: candidateTrimCharacters)
            guard value.count >= 3, value.count <= 80 else { return }
            let key = value.lowercased()
            guard !seen.contains(key) else { return }
            seen.insert(key)
            out.append(value)
        }

        func flushPhrase() {
            defer { phrase.removeAll(keepingCapacity: true) }
            guard !phrase.isEmpty else { return }
            if (2...3).contains(phrase.count) {
                let phraseValue = phrase.joined(separator: " ")
                if phrase.contains(where: { isSingleScreenCandidateToken($0) }) {
                    add(phraseValue)
                }
            }
            for token in phrase where isSingleScreenCandidateToken(token) {
                add(token)
            }
        }

        for raw in text.components(separatedBy: .whitespacesAndNewlines) {
            let token = raw.trimmingCharacters(in: candidateTrimCharacters)
            guard !token.isEmpty else {
                flushPhrase()
                continue
            }
            if isTitleOrMixedCaseToken(token) {
                phrase.append(token)
            } else {
                flushPhrase()
                if isTechnicalScreenCandidateToken(token) {
                    add(token)
                }
            }
            if out.count >= maxVisibleCandidateCount { break }
        }
        flushPhrase()
        if out.count > maxVisibleCandidateCount {
            return Array(out.prefix(maxVisibleCandidateCount))
        }
        return out
    }

    private static let candidateTrimCharacters = CharacterSet(charactersIn: " ,.!?;:()[]{}<>\"'`“”‘’")

    private static let commonScreenCandidateWords: Set<String> = [
        "First", "Second", "Third", "Fourth", "Fifth", "Point", "Text",
        "Start", "End", "Today", "Tomorrow", "Yesterday",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December",
        "The", "This", "That", "These", "Those", "There", "Here", "Okay", "Ok",
        "Reminder", "Calendar", "Message", "Messages",
    ]

    private static func isTitleOrMixedCaseToken(_ token: String) -> Bool {
        isSingleScreenCandidateToken(token)
    }

    private static func isSingleScreenCandidateToken(_ token: String) -> Bool {
        guard token.count >= 4 else { return false }
        guard !commonScreenCandidateWords.contains(token) else { return false }
        guard token.rangeOfCharacter(from: .letters) != nil else { return false }
        guard !isLikelyScreenCandidateOCRJunk(token) else { return false }
        let letters = token.filter { $0.isLetter }
        guard !letters.isEmpty else { return false }
        if letters.allSatisfy({ $0.isUppercase }) && (2...10).contains(letters.count) { return true }
        let letterString = String(letters)
        if isSimpleCapitalizedToken(letterString) { return true }
        if isTwoPartCamelOrAcronymToken(letterString) { return true }
        return false
    }

    private static func isTechnicalScreenCandidateToken(_ token: String) -> Bool {
        let clean = token.trimmingCharacters(in: candidateTrimCharacters)
        guard (4...80).contains(clean.count) else { return false }
        let patterns = [
            "^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)+(?:\\.[a-z0-9]{1,8})?$",
            "^[a-z][a-z0-9]*(?:\\.[a-z0-9]{1,8})$",
            "^[A-Z]{2,}\\d{1,6}$",
        ]
        return patterns.contains { pattern in
            guard let regex = try? NSRegularExpression(pattern: pattern) else { return false }
            let range = NSRange(clean.startIndex..., in: clean)
            return regex.firstMatch(in: clean, range: range) != nil
        }
    }

    private static func isLikelyScreenCandidateOCRJunk(_ token: String) -> Bool {
        let clean = token.trimmingCharacters(in: candidateTrimCharacters)
        let hasLetter = clean.contains { $0.isLetter }
        let hasDigit = clean.contains { $0.isNumber }
        let hasIdentifierSeparator = clean.contains("_") || clean.contains(".") || clean.contains("/") || clean.contains("-")
        if hasLetter && hasDigit && !hasIdentifierSeparator {
            return true
        }
        return false
    }

    private static func isSimpleCapitalizedToken(_ letters: String) -> Bool {
        guard let first = letters.first, first.isUppercase else { return false }
        let rest = letters.dropFirst()
        return letters.count >= 3 && rest.allSatisfy { $0.isLowercase }
    }

    private static func isTwoPartCamelOrAcronymToken(_ letters: String) -> Bool {
        let patterns = [
            "^[A-Z][a-z]{2,}[A-Z][a-z]{2,}$",
            "^[A-Z][a-z]{2,}[A-Z]{2,4}$",
        ]
        return patterns.contains { pattern in
            guard let regex = try? NSRegularExpression(pattern: pattern) else { return false }
            let range = NSRange(letters.startIndex..., in: letters)
            return regex.firstMatch(in: letters, range: range) != nil
        }
    }
}
