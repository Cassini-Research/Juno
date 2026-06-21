import AppKit
import SwiftUI

// MARK: - Broker JSON (transform)

/// Parsed body of `POST /api/broker/session/transform` JSON.
enum JunoBrokerSessionResponse: Equatable {
    case replacementText(String)
    case noteSaved(title: String?, noteId: String?)
    case completed
    case brokerError(code: String, message: String)
    case unreadableBody(String?)

    static func parse(_ data: Data) -> JunoBrokerSessionResponse {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .unreadableBody(String(data: data, encoding: .utf8))
        }
        if let ok = obj["ok"] as? Bool, ok == false {
            let code = (obj["error_code"] as? String) ?? (obj["error"] as? String) ?? "request_failed"
            let msg = (obj["message"] as? String) ?? ""
            let detail = msg.isEmpty ? Self.friendlyMessage(forErrorCode: code) : msg
            return .brokerError(code: code, message: detail)
        }
        if let result = obj["result"] as? [String: Any] {
            if let rep = result["replacement_text"] as? String, !rep.isEmpty {
                return .replacementText(rep)
            }
            if let noteId = result["note_id"] as? String {
                let title = result["title"] as? String
                return .noteSaved(title: title, noteId: noteId)
            }
            if let title = result["title"] as? String, !title.isEmpty {
                return .noteSaved(title: title, noteId: nil)
            }
        }
        return .completed
    }

    private static func friendlyMessage(forErrorCode code: String) -> String {
        switch code {
        case "request_failed", "broker_unreachable":
            return "Could not reach the voice engine. Make sure Juno is running and try again."
        case "surface_denied", "surface_tier_denied":
            return "This action isn’t available for the current app or surface."
        case "utterance_in_progress":
            return "Finish dictation before running this action."
        default:
            return "Something went wrong. Try again in a moment."
        }
    }
}

// MARK: - Window configuration

enum JunoShellBrokerFlow {
    case transformFromClipboard
    case transformSelection

    fileprivate var successTitle: String {
        switch self {
        case .transformFromClipboard: "Polished text"
        case .transformSelection: "Selection rewrite"
        }
    }

    fileprivate var windowTitle: String {
        switch self {
        case .transformFromClipboard: "Juno · Polished text"
        case .transformSelection: "Juno · Selection rewrite"
        }
    }

    fileprivate var failureWindowTitle: String {
        switch self {
        case .transformFromClipboard: "Juno · Transform"
        case .transformSelection: "Juno · Selection rewrite"
        }
    }
}

// MARK: - Model

@MainActor
final class JunoSessionResultModel: ObservableObject {
    enum Tone {
        case success
        case guidance
        case error
    }

    let tone: Tone
    let headerTitle: String
    let headerSubtitle: String?
    /// Main scrollable content (polished text, or extra detail).
    let body: String?
    let frontmostPid: pid_t
    let windowTitle: String
    @Published var copyConfirmed: Bool = false

    var showsCopyButton: Bool {
        guard let body, !body.isEmpty else { return false }
        return tone != .error
    }

    var showsInsertButton: Bool {
        tone == .success && showsCopyButton && frontmostPid > 0
    }

    init(
        tone: Tone,
        headerTitle: String,
        headerSubtitle: String?,
        body: String?,
        frontmostPid: pid_t,
        windowTitle: String
    ) {
        self.tone = tone
        self.headerTitle = headerTitle
        self.headerSubtitle = headerSubtitle
        self.body = body
        self.frontmostPid = frontmostPid
        self.windowTitle = windowTitle
    }
}

// MARK: - View

private struct JunoSessionResultView: View {
    @ObservedObject var model: JunoSessionResultModel
    let onDismiss: () -> Void

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                JunoChromeBrandHeader(style: .compact, title: model.headerTitle, subtitle: nil)
                if let sub = model.headerSubtitle, !sub.isEmpty {
                    Text(sub)
                        .font(.callout)
                        .foregroundStyle(
                            model.tone == .error
                                ? JunoDesignTokens.danger
                                : JunoTheme.secondaryText(scheme)
                        )
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 8)

            if let body = model.body, !body.isEmpty {
                ScrollView {
                    Text(body)
                        .font(.body)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                        .multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 120, maxHeight: 280)
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(JunoTheme.cardBackground(scheme))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .strokeBorder(JunoTheme.subtleBorder(scheme), lineWidth: 1)
                        )
                )
                .padding(.horizontal, 20)
                .padding(.top, 14)
            } else if model.tone == .guidance || model.tone == .error {
                Spacer().frame(height: 8)
            }

            HStack(spacing: 10) {
                Button("Done") { onDismiss() }
                    .keyboardShortcut(.escape)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))

                Spacer()

                if model.tone != .error, model.showsCopyButton {
                    if model.copyConfirmed {
                        Label("Copied", systemImage: "checkmark")
                            .font(.callout)
                            .foregroundStyle(JunoDesignTokens.meadow)
                            .transition(.opacity)
                    } else {
                        Button("Copy") {
                            if let t = model.body { Clipboard.writeString(t) }
                            withAnimation { model.copyConfirmed = true }
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.8) {
                                withAnimation { model.copyConfirmed = false }
                            }
                        }
                        .keyboardShortcut("c", modifiers: .command)
                    }
                }

                if model.showsInsertButton {
                    Button("Paste") {
                        Self.pasteBodyThenDismiss(model: model)
                    }
                    .keyboardShortcut(.return, modifiers: .command)
                    .controlSize(.regular)
                    .buttonStyle(JunoPrimaryActionButtonStyle())
                    .junoNoFocusRing()
                }
            }
            .padding(20)
            .padding(.top, model.body == nil || model.body?.isEmpty == true ? 8 : 4)
        }
        .frame(minWidth: 440, idealWidth: 480)
        .background(JunoTheme.windowBackground(scheme))
    }

    @MainActor
    private static func pasteBodyThenDismiss(model: JunoSessionResultModel) {
        guard let text = model.body, !text.isEmpty else { return }
        let pid = model.frontmostPid
        JunoSessionResultWindow.dismissForPaste()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
            if pid > 0, let app = NSRunningApplication(processIdentifier: pid) {
                app.activate(options: .activateIgnoringOtherApps)
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                // verifyLanded: without read-back this always reports success
                // (the low-level path only confirms the keystroke posted), so a
                // failed paste would dismiss the window AND revert the clipboard,
                // silently losing the result. Verify so restoreAfterFailedPaste
                // re-shows the window and re-arms the clipboard on a real miss.
                let ok = Clipboard.undoSafePaste(text, verifyLanded: true)
                if !ok {
                    JunoSessionResultWindow.restoreAfterFailedPaste(model: model)
                }
            }
        }
    }
}

// MARK: - Window host

private final class JunoSessionResultCloseDelegate: NSObject, NSWindowDelegate {
    let onClose: () -> Void
    init(onClose: @escaping () -> Void) { self.onClose = onClose }
    func windowWillClose(_ notification: Notification) { onClose() }
}

enum JunoSessionResultWindow {
    private static var windowController: NSWindowController?
    private static var closeDelegate: JunoSessionResultCloseDelegate?
    private static var currentModel: JunoSessionResultModel?
    /// Pasteboard polling — used to auto-dismiss the result panel the
    /// moment the user copies the body text via ⌘C anywhere on the
    /// system. NSPasteboard has no native change notification so we
    /// poll changeCount cheaply (~250ms) only while the panel is open.
    private static var pasteboardPollTimer: Timer?
    private static var pasteboardBaselineCount: Int = 0

    /// Preview-first: show result; user copies or pastes into the prior frontmost app.
    @MainActor
    static func presentBrokerResponse(_ response: JunoBrokerSessionResponse, flow: JunoShellBrokerFlow, frontmostPid: pid_t) {
        switch response {
        case .replacementText(let text):
            let model = JunoSessionResultModel(
                tone: .success,
                headerTitle: flow.successTitle,
                headerSubtitle: "Review below, then copy or paste into your document.",
                body: text,
                frontmostPid: frontmostPid,
                windowTitle: flow.windowTitle
            )
            show(model: model, windowTitle: flow.windowTitle)
        case .noteSaved:
            presentTransformUnexpectedSuccess(response: response, frontmostPid: frontmostPid)
        case .completed:
            presentTransformUnexpectedSuccess(response: response, frontmostPid: frontmostPid)
        case .brokerError(_, let message):
            let model = JunoSessionResultModel(
                tone: .error,
                headerTitle: "Couldn’t complete",
                headerSubtitle: message,
                body: nil,
                frontmostPid: frontmostPid,
                windowTitle: flow.failureWindowTitle
            )
            show(model: model, windowTitle: flow.failureWindowTitle)
        case .unreadableBody(let raw):
            let hint = raw.map { $0.count > 400 ? String($0.prefix(400)) + "…" : $0 } ?? ""
            let model = JunoSessionResultModel(
                tone: .error,
                headerTitle: "Unexpected response",
                headerSubtitle: "The voice engine returned data Juno couldn’t read. If this keeps happening, update Juno.",
                body: hint.isEmpty ? nil : hint,
                frontmostPid: frontmostPid,
                windowTitle: flow.failureWindowTitle
            )
            show(model: model, windowTitle: flow.failureWindowTitle)
        }
    }

    @MainActor
    static func presentTransportFailure(flow: JunoShellBrokerFlow, localizedDescription: String, frontmostPid: pid_t) {
        let model = JunoSessionResultModel(
            tone: .error,
            headerTitle: "Couldn’t reach Juno",
            headerSubtitle: localizedDescription,
            body: "Make sure the voice engine is running and try again.",
            frontmostPid: frontmostPid,
            windowTitle: flow.failureWindowTitle
        )
        show(model: model, windowTitle: flow.failureWindowTitle)
    }

    @MainActor
    static func presentTransformEmptyClipboardGuidance() {
        let model = JunoSessionResultModel(
            tone: .guidance,
            headerTitle: "Transform",
            headerSubtitle: "Copy the text you want to polish, then run Transform again. Juno reads from the clipboard and sends it to the voice engine.",
            body: nil,
            frontmostPid: 0,
            windowTitle: "Juno · Transform"
        )
        show(model: model, windowTitle: "Juno · Transform")
    }

    @MainActor
    static func presentTransformEmptySelectionGuidance() {
        let model = JunoSessionResultModel(
            tone: .guidance,
            headerTitle: "Selection rewrite",
            headerSubtitle: "Highlight the text you want to rewrite, then run Selection rewrite again. Juno reads the active selection directly from the current app.",
            body: nil,
            frontmostPid: 0,
            windowTitle: "Juno · Selection rewrite"
        )
        show(model: model, windowTitle: "Juno · Selection rewrite")
    }

    /// Unexpected success shape for transform (e.g. only `completed`).
    @MainActor
    static func presentTransformUnexpectedSuccess(response: JunoBrokerSessionResponse, frontmostPid: pid_t) {
        let body: String?
        let sub: String
        switch response {
        case .noteSaved(let title, let noteId):
            sub = "The engine responded as a save, not as edited text."
            var lines: [String] = []
            if let title, !title.isEmpty { lines.append("Title: \(title)") }
            if let noteId { lines.append("Reference: \(noteId)") }
            body = lines.isEmpty ? nil : lines.joined(separator: "\n")
        case .completed:
            sub = "The request finished, but no edited text was returned."
            body = nil
        case .unreadableBody(let s):
            sub = s ?? ""
            body = nil
        default:
            sub = "Try again or check History in Juno."
            body = nil
        }
        let model = JunoSessionResultModel(
            tone: .guidance,
            headerTitle: "Transform",
            headerSubtitle: sub,
            body: body,
            frontmostPid: frontmostPid,
            windowTitle: "Juno · Transform"
        )
        show(model: model, windowTitle: "Juno · Transform")
    }

    @MainActor
    private static func show(model: JunoSessionResultModel, windowTitle: String) {
        dismiss()
        currentModel = model
        let view = JunoSessionResultView(model: model) { dismiss() }
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.title = windowTitle
        window.styleMask = [.titled, .closable, .resizable]
        window.setContentSize(NSSize(width: 500, height: 360))
        window.center()
        window.isReleasedWhenClosed = false
        window.level = .floating
        let del = JunoSessionResultCloseDelegate {
            windowController = nil
            closeDelegate = nil
            currentModel = nil
        }
        closeDelegate = del
        window.delegate = del
        let wc = NSWindowController(window: window)
        windowController = wc
        wc.showWindow(nil)
        JunoWindowActivation.bringToFront(window)
        startPasteboardWatcher()
    }

    @MainActor
    static func dismiss() {
        stopPasteboardWatcher()
        windowController?.close()
        windowController = nil
        closeDelegate = nil
        currentModel = nil
    }

    /// Called after Paste so `undoSafePaste` is not fighting an open window.
    @MainActor
    fileprivate static func dismissForPaste() {
        stopPasteboardWatcher()
        windowController?.close()
        windowController = nil
        closeDelegate = nil
        currentModel = nil
    }

    @MainActor
    fileprivate static func restoreAfterFailedPaste(model: JunoSessionResultModel) {
        if let text = model.body, !text.isEmpty {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) {
                Clipboard.writeString(text)
            }
        }
        model.copyConfirmed = false
        show(model: model, windowTitle: model.windowTitle)
    }

    // MARK: - Pasteboard watcher (auto-dismiss on external ⌘C)

    /// Watch the system pasteboard for the moment the result body lands
    /// on it. When it does — whether via the in-window Copy button or
    /// the user pressing ⌘C on the visible result text — fade the panel
    /// away. Matches the user's mental model: "I copy and the HUD
    /// dismisses".
    @MainActor
    private static func startPasteboardWatcher() {
        stopPasteboardWatcher()
        pasteboardBaselineCount = NSPasteboard.general.changeCount
        let timer = Timer(timeInterval: 0.25, repeats: true) { _ in
            Task { @MainActor in checkPasteboardForBodyAndDismiss() }
        }
        RunLoop.main.add(timer, forMode: .common)
        pasteboardPollTimer = timer
    }

    @MainActor
    private static func stopPasteboardWatcher() {
        pasteboardPollTimer?.invalidate()
        pasteboardPollTimer = nil
    }

    @MainActor
    private static func checkPasteboardForBodyAndDismiss() {
        guard let model = currentModel,
              let body = model.body, !body.isEmpty,
              windowController != nil else {
            stopPasteboardWatcher()
            return
        }
        let pb = NSPasteboard.general
        guard pb.changeCount != pasteboardBaselineCount else { return }
        pasteboardBaselineCount = pb.changeCount
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if let s = pb.string(forType: .string)?.trimmingCharacters(in: .whitespacesAndNewlines),
           s == trimmed {
            // Brief beat so the user perceives the copy landed.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.20) { dismiss() }
        }
    }
}
