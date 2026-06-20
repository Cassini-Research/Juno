// JunoPasteTargetDispositionTests.swift
//
// JunoTargetApplicationTracker.pasteTargetDisposition decides how the app
// sampled as frontmost at a paste boundary should affect the paste target.
//
// Regression context (the "second press targets the wrong app / nothing
// pasted" report): Juno's floating HUD panel is canBecomeKey while visible, so
// it can be momentarily frontmost when finalize samples NSWorkspace's
// frontmost app — even though the user is dictating into another app (e.g.
// Brave). PR #19 made that case gate the paste OFF (likelyPasteDestination =
// false), which mis-filed the result into the copy-ready overlay instead of
// pasting into the real target. An ignored/own surface must therefore resolve
// to .preservePrior (keep the start-captured target; do NOT gate off), while a
// genuinely-focused non-editable system pane (System Settings) stays .copyOnly.

import XCTest
@testable import JunoShell

final class JunoPasteTargetDispositionTests: XCTestCase {
    private typealias Tracker = JunoTargetApplicationTracker

    func testNotificationCenterPreservesPriorTarget() {
        // Regression: must NOT be .copyOnly — a transient ignored surface at
        // finalize should keep the real target, not divert to copy.
        XCTAssertEqual(
            Tracker.pasteTargetDisposition(bundleId: "com.apple.notificationcenterui", name: nil),
            .preservePrior
        )
    }

    func testControlCenterPreservesPriorTarget() {
        XCTAssertEqual(
            Tracker.pasteTargetDisposition(bundleId: "com.apple.controlcenter", name: "Control Center"),
            .preservePrior
        )
    }

    func testSystemSettingsIsCopyOnly() {
        XCTAssertEqual(
            Tracker.pasteTargetDisposition(bundleId: "com.apple.systempreferences", name: "System Settings"),
            .copyOnly
        )
        // Legacy bundle id / name still classified as copy-only.
        XCTAssertEqual(
            Tracker.pasteTargetDisposition(bundleId: nil, name: "System Preferences"),
            .copyOnly
        )
    }

    func testRealAppsAreAdopted() {
        for bid in ["com.brave.Browser", "com.google.Chrome", "com.apple.Safari", "com.apple.Notes"] {
            XCTAssertEqual(
                Tracker.pasteTargetDisposition(bundleId: bid, name: nil),
                .adopt,
                "expected \(bid) to be adopted as the paste target"
            )
        }
    }

    func testCaseAndWhitespaceInsensitiveSystemPaneMatch() {
        XCTAssertTrue(Tracker.isNonEditableSystemPasteSurface(bundleId: "  COM.APPLE.SystemPreferences ", name: nil))
        XCTAssertFalse(Tracker.isNonEditableSystemPasteSurface(bundleId: "com.brave.Browser", name: nil))
    }
}
