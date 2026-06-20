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

    func testFnSelectionDisablesSystemEmojiActionAndBacksUpExistingValue() {
        XCTAssertEqual(
            JunoFnGlobeSystemActionPreference.plannedAction(
                shortcut: .fn,
                currentValue: 2,
                didOverride: false,
                backupWasPresent: false,
                backupValue: nil
            ),
            .disableEmoji(backup: .init(wasPresent: true, value: 2))
        )
    }

    func testFnSelectionBacksUpAbsentSystemValue() {
        XCTAssertEqual(
            JunoFnGlobeSystemActionPreference.plannedAction(
                shortcut: .fn,
                currentValue: nil,
                didOverride: false,
                backupWasPresent: false,
                backupValue: nil
            ),
            .disableEmoji(backup: .init(wasPresent: false, value: nil))
        )
    }

    func testFnSelectionKeepsExistingBackupWhenReapplyingOverride() {
        XCTAssertEqual(
            JunoFnGlobeSystemActionPreference.plannedAction(
                shortcut: .fn,
                currentValue: 2,
                didOverride: true,
                backupWasPresent: true,
                backupValue: 1
            ),
            .disableEmoji(backup: nil)
        )
    }

    func testFnSelectionDoesNothingWhenSystemActionAlreadyDisabled() {
        XCTAssertEqual(
            JunoFnGlobeSystemActionPreference.plannedAction(
                shortcut: .fn,
                currentValue: 0,
                didOverride: false,
                backupWasPresent: false,
                backupValue: nil
            ),
            .none
        )
    }

    func testLeavingFnRestoresPreviousSystemValue() {
        XCTAssertEqual(
            JunoFnGlobeSystemActionPreference.plannedAction(
                shortcut: .rightOption,
                currentValue: 0,
                didOverride: true,
                backupWasPresent: true,
                backupValue: 2
            ),
            .restore(value: 2)
        )
    }

    func testLeavingFnDeletesSystemValueWhenItWasPreviouslyUnset() {
        XCTAssertEqual(
            JunoFnGlobeSystemActionPreference.plannedAction(
                shortcut: .rightOption,
                currentValue: 0,
                didOverride: true,
                backupWasPresent: false,
                backupValue: nil
            ),
            .restore(value: nil)
        )
    }

    func testLeavingFnDoesNothingWithoutJunoOverride() {
        XCTAssertEqual(
            JunoFnGlobeSystemActionPreference.plannedAction(
                shortcut: .rightOption,
                currentValue: 2,
                didOverride: false,
                backupWasPresent: false,
                backupValue: nil
            ),
            .none
        )
    }
}
