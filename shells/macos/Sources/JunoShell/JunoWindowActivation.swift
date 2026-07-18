import AppKit

/// macOS 15 (Sequoia) and later: `NSApplication.activate(ignoringOtherApps:)`
/// is deprecated and no longer reliably raises a normal window from menu-bar
/// / dock / Finder reopen paths. Accessibility can report the window focused
/// while Window Server still keeps another app above it — this is exactly
/// what made Juno "look frozen" while AX automation could still trigger
/// buttons (the audit's #1 user-reported click-eat bug).
///
/// Use these helpers in place of the deprecated APIs so every window the
/// shell opens (main, onboarding, launch splash, broker help, diagnostics,
/// session result, action-permission flows) goes through the same proven
/// raise sequence.
@MainActor
enum JunoWindowActivation {

    /// Bring the given window to the foreground reliably on macOS 15+.
    ///
    /// Critical for MenuBarExtra-based apps (which Juno is): on Sequoia,
    /// windows opened from a `MenuBarExtra` app silently fail to become
    /// key/main even when activation policy is `.regular`. The window
    /// renders, AX automation can target buttons, but the responder chain
    /// is never installed — every mouse click is dropped. This is the
    /// user-reported "first launch fine, relaunch broken" symptom.
    ///
    /// Workaround sequence (proven for MenuBarExtra apps on Sequoia, see
    /// docstring of this enum):
    ///   1. Force `.regular` activation policy. macOS will not let a
    ///      window become key while the app is `.accessory`. Even when
    ///      the user prefers a Dock-less app, we must temporarily flip to
    ///      `.regular` so the window can receive clicks. The application
    ///      delegate restores the user's preferred policy when the window
    ///      closes (see `JunoMainWindowCloseDelegate`).
    ///   2. `NSApp.activate()` — modern no-arg API.
    ///   3. `window.deminiaturize(nil)` — un-minimize if minimized.
    ///   4. `window.orderFrontRegardless()` — bypass Apple's app-
    ///      activation gate; needed when the app didn't actually become
    ///      frontmost.
    ///   5. `window.makeKeyAndOrderFront(nil)` — install first-responder
    ///      chain so SwiftUI buttons actually receive clicks.
    ///   6. Follow-up `NSApp.activate()` on next runloop tick — this is
    ///      the Sequoia-specific belt-and-suspenders. Without it, the
    ///      window has key status but the app still isn't frontmost in
    ///      Window Server's eyes, which silently drops mouse events.
    ///
    /// Safe to call on a window that is already visible / key — the
    /// calls are idempotent.
    static func bringToFront(_ window: NSWindow) {
        // Step 1: ensure the app is eligible to own a key window.
        // MenuBarExtra apps in `.accessory` mode silently can't.
        if NSApp.activationPolicy() != .regular {
            NSApp.setActivationPolicy(.regular)
        }
        var behavior = window.collectionBehavior
        behavior.insert(.moveToActiveSpace)
        window.collectionBehavior = behavior
        NSApp.unhide(nil)
        // Steps 2-5: standard raise sequence.
        //
        // CRITICAL: use the deprecated `activate(ignoringOtherApps:)` API,
        // not the new no-arg `NSApp.activate()`. On macOS Sequoia the new
        // API has stricter "permitted to activate" requirements that can
        // silently no-op when called from a programmatic relaunch path
        // (the user-input-event association expires before our handler
        // runs). The deprecated API forces activation unconditionally and
        // is the only one that reliably brings the window forward on
        // relaunch via `open /Applications/Juno.app` on Sequoia.
        //
        // Empirically: builds from 2026-05-06 using the deprecated API
        // worked. The 2026-05-07 commit 37b5f60 replaced it with the new
        // API based on the assumption that "deprecated" meant "doesn't
        // work" — which broke relaunch activation. The deprecation
        // warning is about API style, not runtime behavior; Apple has
        // not removed the deprecated method.
        @available(macOS, deprecated: 14.0)
        func _activateLegacy() {
            NSApp.activate(ignoringOtherApps: true)
        }
        NSApp.activate()
        NSRunningApplication.current.activate(options: [.activateAllWindows])
        _activateLegacy()
        window.deminiaturize(nil)
        window.orderFrontRegardless()
        window.makeKeyAndOrderFront(nil)
        // Step 6: belt-and-suspenders — invoke both APIs on the next
        // runloop tick. If the deprecated API succeeded, the no-arg
        // call is a harmless no-op. If Apple ever fully removes the
        // deprecated API in a future macOS, the no-arg call still runs.
        DispatchQueue.main.async {
            NSApp.unhide(nil)
            NSApp.activate()
            NSRunningApplication.current.activate(options: [.activateAllWindows])
            _activateLegacy()
            window.orderFrontRegardless()
            window.makeKeyAndOrderFront(nil)
        }
    }

    /// Activate the application without targeting a specific window.
    /// Use only when there is no specific window to raise (e.g. you're
    /// about to create one, or asking the system to grant focus to a
    /// permission dialog about to appear). For "make this existing
    /// window front", always use ``bringToFront(_:)``.
    ///
    /// Also forces `.regular` activation policy for the same reason as
    /// `bringToFront`: MenuBarExtra apps in `.accessory` mode do not
    /// reliably receive activation on Sequoia. The app stays `.regular`
    /// after the call; `JunoDockVisibility.applyCurrent()` restores the
    /// user's `showInDock` preference when the visible window closes.
    static func activateApp() {
        if NSApp.activationPolicy() != .regular {
            NSApp.setActivationPolicy(.regular)
        }
        // Use deprecated API for the same Sequoia reason as bringToFront.
        @available(macOS, deprecated: 14.0)
        func _activateLegacy() {
            NSApp.activate(ignoringOtherApps: true)
        }
        NSApp.unhide(nil)
        NSApp.activate()
        NSRunningApplication.current.activate(options: [.activateAllWindows])
        _activateLegacy()
    }
}

final class JunoActivationRestoringWindowDelegate: NSObject, NSWindowDelegate {
    private let onClose: () -> Void

    init(onClose: @escaping () -> Void) {
        self.onClose = onClose
    }

    func windowWillClose(_ notification: Notification) {
        onClose()
        if JunoUserDefaults.menuBarOnlyModeEnabled {
            NSApp.setActivationPolicy(.accessory)
        }
    }
}
