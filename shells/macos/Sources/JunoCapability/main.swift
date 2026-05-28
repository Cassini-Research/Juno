// juno-capability
//
// One-shot Accessibility probe. Prints a single JSON line describing the
// current frontmost app, focused UI element, and whether the focused
// field is a secure-text input (password / PIN). The broker uses this
// to refuse dictation when:
//
//   - the focused element is a secure text field (subrole
//     AXSecureTextField), regardless of app;
//   - the frontmost app is on a managed/banking blocklist;
//   - the helper does not have Accessibility trust (user hasn't granted
//     the permission yet).
//
// Why a separate helper:
//   The JunoShell menu-bar app can do the same AX probes in-process,
//   but running the probe as a fresh, short-lived binary isolates its
//   failures (null UIElement, stale AX cache after app switching,
//   Accessibility permission race on first run) from the long-running
//   shell event loop. It also makes the probe easy to script from the
//   broker: the Python side just spawns the binary with a timeout.
//
// Stdout protocol (a single JSON object, no trailing newline required):
//
//   {
//     "ok": true,
//     "has_ax_trust": true,
//     "frontmost_app_bundle_id": "com.apple.Safari",
//     "frontmost_app_name": "Safari",
//     "app_name": "Safari",            // alias; matches the python context provider
//     "app_bundle_id": "com.apple.Safari",
//     "frontmost_pid": 1234,
//     "window_title": "Sign in — Example Bank",
//     "focused_role": "AXTextField",
//     "focused_subrole": "AXSecureTextField",
//     "focused_is_secure": true,
//     "selected_text": "",             // AXSelectedText of the focused element
//     "focused_text_before": "Hi team, ", // caret-left slice (<= 240 chars)
//     "focused_text_after": " how are you?", // caret-right slice (<= 240 chars)
//     "clipboard_text": "latest copied string (<= 240 chars)"
//   }
//
//   The context fields (selected_text / focused_text_* / clipboard_text)
//   are best-effort: many apps don't expose AXValue for focused text, or
//   redact clipboard for security. We emit empty strings rather than
//   omitting the keys so the python side can treat the schema as stable.
//
//   On failure ("ok": false) we include "error": "<human readable>"
//   and still set any fields that were readable.
//
// Exit code:
//   0 always (the broker interprets the JSON). Non-zero is reserved for
//   catastrophic startup failure (can't start Cocoa), never for a
//   "secure field detected" result.

import ApplicationServices
import Cocoa
import Foundation

private func postPasteShortcut() -> Int32 {
    guard AXIsProcessTrusted() else {
        FileHandle.standardError.write(Data("juno-capability: accessibility not trusted for paste\n".utf8))
        return 2
    }

    let vKey: CGKeyCode = 0x09
    let source = CGEventSource(stateID: .privateState)
    guard
        let down = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: true),
        let up = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: false)
    else {
        FileHandle.standardError.write(Data("juno-capability: CGEvent creation failed\n".utf8))
        return 3
    }

    down.flags = .maskCommand
    up.flags = .maskCommand
    down.post(tap: .cgSessionEventTap)
    usleep(8_000)
    up.post(tap: .cgSessionEventTap)
    usleep(30_000)
    return 0
}

// MARK: - JSON helpers

// Hard cap on any free-text field we copy off another app: clipboard,
// selected text, caret-context. Keep enough local context for the writer
// to repair messy dictation, while Python ContextPlane and writer packets
// still enforce tighter prompt budgets before model calls.
private let kMaxContextFieldLen = 1600

final class Capability {
    var hasAxTrust = false
    var frontmostAppBundleId: String?
    var frontmostAppName: String?
    var frontmostPid: Int?
    var windowTitle: String?
    var focusedRole: String?
    var focusedSubrole: String?
    var focusedIsSecure = false
    var selectedText: String?
    var focusedTextBefore: String?
    var focusedTextAfter: String?
    var clipboardText: String?
    // Path (or URL) of the document the focused window is editing,
    // when the app exposes it via AXDocument. For IDEs and editors
    // this is the absolute file path; for web apps it can be a URL.
    // The broker uses it as a bias hint ("the user is editing
    // foo.py, expect python identifiers") and as a tag on the
    // per-utterance trace so history views can group by file.
    var focusedDocumentPath: String?
    var error: String?

    func toJSON() -> String {
        var dict: [String: Any] = [
            "ok": error == nil,
            "has_ax_trust": hasAxTrust,
            "focused_is_secure": focusedIsSecure,
            // Stable shape: downstream code expects these keys to
            // always exist (even as empty strings) so it can skip
            // a branch per field.
            "selected_text": selectedText ?? "",
            "focused_text_before": focusedTextBefore ?? "",
            "focused_text_after": focusedTextAfter ?? "",
            "clipboard_text": clipboardText ?? "",
            // BCP-47-ish tag for Python ITN (date / clock / currency decimals).
            "locale_identifier": Locale.current.identifier,
        ]
        if let v = frontmostAppBundleId {
            dict["frontmost_app_bundle_id"] = v
            // Alias matching ``MacOSDesktopContextProvider._helper_payload``;
            // we emit both so the provider works without any schema
            // translation.
            dict["app_bundle_id"] = v
        }
        if let v = frontmostAppName {
            dict["frontmost_app_name"] = v
            dict["app_name"] = v
        }
        if let v = frontmostPid { dict["frontmost_pid"] = v }
        if let v = windowTitle { dict["window_title"] = v }
        if let v = focusedRole { dict["focused_role"] = v }
        if let v = focusedSubrole { dict["focused_subrole"] = v }
        if let v = focusedDocumentPath {
            dict["focused_document_path"] = v
        }
        if let v = error { dict["error"] = v }
        do {
            let data = try JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])
            return String(data: data, encoding: .utf8) ?? "{\"ok\":false,\"error\":\"json_utf8_failed\"}"
        } catch {
            FileHandle.standardError.write(Data("juno-capability: JSON serialization failed: \(error)\n".utf8))
            return "{\"ok\":false,\"error\":\"json_serialize_failed\"}"
        }
    }
}

@inline(__always)
private func clip(_ s: String?) -> String? {
    guard let s, !s.isEmpty else { return s }
    if s.count <= kMaxContextFieldLen { return s }
    return String(s.prefix(kMaxContextFieldLen))
}

@inline(__always)
func axString(_ element: AXUIElement, _ attr: CFString) -> String? {
    var value: CFTypeRef?
    let err = AXUIElementCopyAttributeValue(element, attr, &value)
    guard err == .success, let v = value as? String else { return nil }
    return v
}

@inline(__always)
func axInt(_ element: AXUIElement, _ attr: CFString) -> Int? {
    var value: CFTypeRef?
    let err = AXUIElementCopyAttributeValue(element, attr, &value)
    guard err == .success else { return nil }
    if let n = value as? NSNumber { return n.intValue }
    return nil
}

// Read a substring of a text element by CFRange. Used to pull text
// immediately before / after the caret without copying the whole
// document into the capability JSON.
func axRangeString(_ element: AXUIElement, location: Int, length: Int) -> String? {
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

@inline(__always)
func axElement(_ element: AXUIElement, _ attr: CFString) -> AXUIElement? {
    var value: CFTypeRef?
    let err = AXUIElementCopyAttributeValue(element, attr, &value)
    guard err == .success else { return nil }
    // CFGetTypeID comparison would be safer; in practice AX returns the
    // correct type or an error. Force-cast via CFTypeRef -> AXUIElement.
    if CFGetTypeID(value) == AXUIElementGetTypeID() {
        // swiftlint:disable:next force_cast
        return (value as! AXUIElement)
    }
    return nil
}

func probe() -> Capability {
    let cap = Capability()

    // AXIsProcessTrusted returns true when the current binary has been
    // granted Accessibility permission. We *don't* prompt here — the
    // JunoShell app handles prompting once, at its own discretion.
    cap.hasAxTrust = AXIsProcessTrusted()

    guard let frontmost = NSWorkspace.shared.frontmostApplication else {
        cap.error = "no_frontmost_app"
        return cap
    }
    cap.frontmostAppBundleId = frontmost.bundleIdentifier
    cap.frontmostAppName = frontmost.localizedName
    cap.frontmostPid = Int(frontmost.processIdentifier)

    guard cap.hasAxTrust else {
        // Without AX trust we can still report the app; focused-element
        // queries will silently fail. That's enough for a decision
        // ("ask user to grant permission"); the broker surfaces this.
        cap.error = "ax_permission_not_granted"
        return cap
    }

    let appElement = AXUIElementCreateApplication(frontmost.processIdentifier)

    // Window title — best-effort, focused window first, else main window.
    // While we're here, also read ``kAXDocumentAttribute`` which macOS
    // document-based apps (Xcode, TextEdit, Pages, Preview, Mail
    // compose, many IDEs) expose as the path/URL of the open
    // document. The attribute is string-typed even for file paths
    // ("/Users/example-user/x.py") and URL-typed for Safari/Chrome (which
    // we skip — context provider will fall back to window title
    // parsing for those).
    if let focusedWindow = axElement(appElement, kAXFocusedWindowAttribute as CFString) {
        cap.windowTitle = axString(focusedWindow, kAXTitleAttribute as CFString)
        if let docPath = axString(focusedWindow, kAXDocumentAttribute as CFString),
           !docPath.isEmpty {
            cap.focusedDocumentPath = clip(docPath)
        }
    }
    if cap.windowTitle == nil,
       let mainWindow = axElement(appElement, kAXMainWindowAttribute as CFString) {
        cap.windowTitle = axString(mainWindow, kAXTitleAttribute as CFString)
        if cap.focusedDocumentPath == nil,
           let docPath = axString(mainWindow, kAXDocumentAttribute as CFString),
           !docPath.isEmpty {
            cap.focusedDocumentPath = clip(docPath)
        }
    }

    // Focused UI element — this is the one we care about for the
    // secure-field check.
    if let focused = axElement(appElement, kAXFocusedUIElementAttribute as CFString) {
        cap.focusedRole = axString(focused, kAXRoleAttribute as CFString)
        cap.focusedSubrole = axString(focused, kAXSubroleAttribute as CFString)
        if cap.focusedSubrole == "AXSecureTextField" {
            cap.focusedIsSecure = true
        }
        // Only try to read text when the field is NOT secure. We
        // never want selected_text / focused_text_* to capture
        // password masks or partial password characters.
        if !cap.focusedIsSecure {
            if let sel = axString(focused, kAXSelectedTextAttribute as CFString),
               !sel.isEmpty {
                cap.selectedText = clip(sel)
            }

            // AXSelectedTextRange lets us slice ``focused_text_before``
            // and ``focused_text_after`` from the value. Many apps
            // (Safari text areas, Mail compose) support this; those
            // that don't will silently skip.
            var selRange = CFRange(location: 0, length: 0)
            var rangeValue: CFTypeRef?
            let rangeErr = AXUIElementCopyAttributeValue(
                focused,
                kAXSelectedTextRangeAttribute as CFString,
                &rangeValue
            )
            if rangeErr == .success,
               let axRangeRef = rangeValue,
               CFGetTypeID(axRangeRef) == AXValueGetTypeID() {
                // swiftlint:disable:next force_cast
                let axRange = axRangeRef as! AXValue
                if AXValueGetType(axRange) == .cfRange
                    && AXValueGetValue(axRange, .cfRange, &selRange) {
                    let caret = max(0, selRange.location)
                    let beforeLen = min(caret, kMaxContextFieldLen)
                    if beforeLen > 0,
                       let before = axRangeString(
                        focused,
                        location: caret - beforeLen,
                        length: beforeLen
                       ), !before.isEmpty {
                        cap.focusedTextBefore = clip(before)
                    }
                    let afterStart = caret + max(0, selRange.length)
                    if let after = axRangeString(
                        focused,
                        location: afterStart,
                        length: kMaxContextFieldLen
                    ), !after.isEmpty {
                        cap.focusedTextAfter = clip(after)
                    }
                }
            }
        }
    }

    // Clipboard — best-effort. We only copy the first text
    // representation and clip aggressively. When the focused element
    // is secure we skip clipboard entirely; a password manager paste
    // may still be on the pasteboard and we don't want to trace it.
    if !cap.focusedIsSecure {
        let pb = NSPasteboard.general
        if let s = pb.string(forType: .string), !s.isEmpty {
            cap.clipboardText = clip(s)
        }
    }

    return cap
}

if CommandLine.arguments.dropFirst().contains("--paste") {
    exit(postPasteShortcut())
}

// NSWorkspace, AXUIElement*, and NSPasteboard all work correctly from a
// plain command-line process without creating an NSApplication instance.
// We intentionally do NOT call NSApplication.shared here: doing so
// registers the binary with the window server as a GUI application, which
// on macOS Sonoma / Sequoia can cause the OS to open a Terminal window
// for each short-lived invocation (juno-capability is spawned every few
// seconds by the surface-polling loop).
let result = probe()
FileHandle.standardOutput.write(Data(result.toJSON().utf8))
exit(0)
