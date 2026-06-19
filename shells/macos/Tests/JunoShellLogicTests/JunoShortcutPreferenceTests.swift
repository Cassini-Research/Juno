import XCTest
@testable import JunoShell

final class JunoShortcutPreferenceTests: XCTestCase {
    func testFnToggleRestartsHotkeyBridgeOnlyAfterOnboarding() {
        XCTAssertFalse(JunoShortcutPreference.shouldRestartHotkeyBridge(
            previous: .rightOption,
            new: .fn,
            onboardingCompleted: false
        ))
        XCTAssertTrue(JunoShortcutPreference.shouldRestartHotkeyBridge(
            previous: .rightOption,
            new: .fn,
            onboardingCompleted: true
        ))
        XCTAssertTrue(JunoShortcutPreference.shouldRestartHotkeyBridge(
            previous: .fn,
            new: .rightOption,
            onboardingCompleted: true
        ))
    }

    func testNonFnTransitionsDoNotRestartHotkeyBridge() {
        XCTAssertFalse(JunoShortcutPreference.shouldRestartHotkeyBridge(
            previous: .rightOption,
            new: .rightCommand,
            onboardingCompleted: true
        ))
        XCTAssertFalse(JunoShortcutPreference.shouldRestartHotkeyBridge(
            previous: .fn,
            new: .fn,
            onboardingCompleted: true
        ))
    }
}
