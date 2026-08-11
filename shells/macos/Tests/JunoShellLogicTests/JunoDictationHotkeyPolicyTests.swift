import Foundation
import XCTest
@testable import JunoShell

final class JunoDictationHotkeyPolicyTests: XCTestCase {
    func testStartupRepeatWithinDebounceIsIgnored() {
        let now: TimeInterval = 10
        let debounceUntil = JunoDictationHotkeyPolicy.startupDebounceUntil(startedAt: now)

        XCTAssertEqual(
            JunoDictationHotkeyPolicy.action(
                for: .checkingCapability,
                now: now + 0.2,
                startupDebounceUntil: debounceUntil
            ),
            .ignoreStartupRepeat
        )
        XCTAssertEqual(
            JunoDictationHotkeyPolicy.action(
                for: .checkingMic,
                now: now + 0.2,
                startupDebounceUntil: debounceUntil
            ),
            .ignoreStartupRepeat
        )
        XCTAssertEqual(
            JunoDictationHotkeyPolicy.action(
                for: .waitingSpeech,
                now: now + 0.2,
                startupDebounceUntil: debounceUntil
            ),
            .ignoreStartupRepeat
        )
    }

    func testStartupStatesResumeNormalMeaningAfterDebounce() {
        let now: TimeInterval = 10
        let debounceUntil = JunoDictationHotkeyPolicy.startupDebounceUntil(startedAt: now)

        XCTAssertEqual(
            JunoDictationHotkeyPolicy.action(
                for: .checkingCapability,
                now: debounceUntil + 0.1,
                startupDebounceUntil: debounceUntil
            ),
            .cancelOpening
        )
        XCTAssertEqual(
            JunoDictationHotkeyPolicy.action(
                for: .checkingMic,
                now: debounceUntil + 0.1,
                startupDebounceUntil: debounceUntil
            ),
            .stop
        )
        XCTAssertEqual(
            JunoDictationHotkeyPolicy.action(
                for: .waitingSpeech,
                now: debounceUntil + 0.1,
                startupDebounceUntil: debounceUntil
            ),
            .stop
        )
    }

    func testActiveListeningCanStopInsideOriginalDebounceWindow() {
        let now: TimeInterval = 10
        let debounceUntil = JunoDictationHotkeyPolicy.startupDebounceUntil(startedAt: now)

        XCTAssertEqual(
            JunoDictationHotkeyPolicy.action(
                for: .listening,
                now: now + 0.2,
                startupDebounceUntil: debounceUntil
            ),
            .stop
        )
    }

    func testTerminalAndRefiningStatesKeepExistingBehavior() {
        XCTAssertEqual(
            JunoDictationHotkeyPolicy.action(
                for: .blocked(reason: .axPermissionMissing),
                now: 10,
                startupDebounceUntil: 11
            ),
            .resetTerminal
        )
        XCTAssertEqual(
            JunoDictationHotkeyPolicy.action(
                for: .error(reason: .micNoAudio),
                now: 10,
                startupDebounceUntil: 11
            ),
            .resetTerminal
        )
        XCTAssertEqual(
            JunoDictationHotkeyPolicy.action(
                for: .refining,
                now: 10,
                startupDebounceUntil: 11
            ),
            .ignore
        )
    }
}
