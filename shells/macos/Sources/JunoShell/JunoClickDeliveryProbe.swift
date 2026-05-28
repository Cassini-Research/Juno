import AppKit
import CoreGraphics
import Foundation

/// Test-only click-delivery probe.
///
/// Enabled only when ``JUNO_UI_TEST_CLICK_PROBE`` is set in the app's
/// environment. The hardware UI smoke script launches Juno with that env var,
/// reads the probe's CGWindow coordinates from the JSON marker, and posts a
/// real CGEvent mouse click at the window center. Accessibility-driven
/// `clickbutton` tests are intentionally insufficient for the PR #66
/// regression: the broken builds accepted AX button clicks while physical
/// Window Server events never reached the app window.
@MainActor
enum JunoClickDeliveryProbe {
    private static var windowController: NSWindowController?

    static var isRequested: Bool {
        guard let raw = ProcessInfo.processInfo.environment["JUNO_UI_TEST_CLICK_PROBE"] else {
            return false
        }
        return !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    static func showIfRequested() {
        guard isRequested else { return }
        guard let markerURL = markerURL else {
            NSLog("Juno UI test probe requested without a usable marker path")
            return
        }

        if let existing = windowController?.window {
            publishReadyWhenClickable(markerURL: markerURL, window: existing, clickCount: nil)
            return
        }

        let view = JunoClickProbeView(markerURL: markerURL)
        let screenFrame = NSScreen.main?.visibleFrame ?? NSRect(x: 120, y: 120, width: 960, height: 640)
        let size = NSSize(width: 360, height: 220)
        let frame = NSRect(
            x: screenFrame.midX - size.width / 2,
            y: screenFrame.midY - size.height / 2,
            width: size.width,
            height: size.height
        )
        let window = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Juno Click Delivery Probe"
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.initialFirstResponder = view

        let controller = NSWindowController(window: window)
        windowController = controller
        controller.showWindow(nil)
        window.makeFirstResponder(view)

        publishReadyWhenClickable(markerURL: markerURL, window: window, clickCount: 0)
    }

    private static var markerURL: URL? {
        guard let raw = ProcessInfo.processInfo.environment["JUNO_UI_TEST_CLICK_PROBE"] else {
            return nil
        }
        let path = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path)
    }

    fileprivate static func writeProbeEvent(
        _ event: String,
        markerURL: URL,
        window: NSWindow?,
        clickCount: Int?
    ) {
        var payload: [String: Any] = [
            "event": event,
            "pid": ProcessInfo.processInfo.processIdentifier,
            "timestamp": Date().timeIntervalSince1970,
            "app_active": NSApp.isActive,
        ]
        if let window {
            payload["window_number"] = window.windowNumber
            payload["window_key"] = window.isKeyWindow
            payload["window_main"] = window.isMainWindow
            if let bounds = cgBounds(for: window) {
                payload["window_x"] = bounds.origin.x
                payload["window_y"] = bounds.origin.y
                payload["window_width"] = bounds.width
                payload["window_height"] = bounds.height
                payload["click_x"] = bounds.midX
                payload["click_y"] = bounds.midY
            }
        }
        if let clickCount {
            payload["click_count"] = clickCount
        }

        do {
            let dir = markerURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: markerURL, options: .atomic)
        } catch {
            NSLog("Juno UI test probe write failed: \(error.localizedDescription)")
        }
    }

    private static func publishReadyWhenClickable(
        markerURL: URL,
        window: NSWindow,
        clickCount: Int?,
        remainingAttempts: Int = 12
    ) {
        JunoWindowActivation.bringToFront(window)
        if let contentView = window.contentView {
            window.makeFirstResponder(contentView)
        }

        let hasBounds = cgBounds(for: window) != nil
        let clickable = NSApp.isActive && window.isKeyWindow && hasBounds
        if clickable {
            writeProbeEvent("ready", markerURL: markerURL, window: window, clickCount: clickCount)
            return
        }

        let event = hasBounds ? "waiting_for_focus" : "waiting_for_window"
        writeProbeEvent(event, markerURL: markerURL, window: window, clickCount: clickCount)
        guard remainingAttempts > 0 else {
            return
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            publishReadyWhenClickable(
                markerURL: markerURL,
                window: window,
                clickCount: clickCount,
                remainingAttempts: remainingAttempts - 1
            )
        }
    }

    private static func cgBounds(for window: NSWindow) -> CGRect? {
        guard let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID)
            as? [[String: Any]]
        else {
            return nil
        }
        for item in list {
            guard let number = item[kCGWindowNumber as String] as? Int, number == window.windowNumber else {
                continue
            }
            guard let bounds = item[kCGWindowBounds as String] as? [String: Any] else {
                return nil
            }
            return CGRect(dictionaryRepresentation: bounds as CFDictionary)
        }
        return nil
    }
}

private final class JunoClickProbeView: NSView {
    private let markerURL: URL
    private var clickCount = 0

    init(markerURL: URL) {
        self.markerURL = markerURL
        super.init(frame: NSRect(x: 0, y: 0, width: 360, height: 220))
        wantsLayer = true
        layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
    }

    required init?(coder: NSCoder) {
        nil
    }

    override var acceptsFirstResponder: Bool { true }

    override func mouseDown(with event: NSEvent) {
        clickCount += 1
        JunoClickDeliveryProbe.writeProbeEvent(
            "clicked",
            markerURL: markerURL,
            window: window,
            clickCount: clickCount
        )
        super.mouseDown(with: event)
    }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.windowBackgroundColor.setFill()
        dirtyRect.fill()
        let title = "Juno click probe"
        let detail = "Hardware UI test target"
        let titleAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 22, weight: .semibold),
            .foregroundColor: NSColor.labelColor,
        ]
        let detailAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 13, weight: .regular),
            .foregroundColor: NSColor.secondaryLabelColor,
        ]
        title.draw(at: NSPoint(x: 28, y: bounds.midY + 6), withAttributes: titleAttrs)
        detail.draw(at: NSPoint(x: 28, y: bounds.midY - 22), withAttributes: detailAttrs)
    }
}
