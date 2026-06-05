import AppKit
import XCTest
@testable import JunoHotkeyCore

final class JunoHotkeyCoreTests: XCTestCase {
    func testNonRepeatCommandCEmitsCopy() {
        let line = JunoHotkeyEventLine.commandCopyLine(
            keyCode: 8,
            modifierFlags: [.command],
            isRepeat: false
        )

        XCTAssertEqual(line, JunoHotkeyEventLine.copy)
        XCTAssertTrue(JunoHotkeyEventLine.isCommandCopy(
            keyCode: 8,
            modifierFlags: [.command],
            isRepeat: false
        ))
        XCTAssertTrue(JunoHotkeyEventLine.isCopyLine("COPY"))
    }

    func testRepeatCommandCReturnsNil() {
        XCTAssertNil(JunoHotkeyEventLine.commandCopyLine(
            keyCode: 8,
            modifierFlags: [.command],
            isRepeat: true
        ))
    }

    func testCommandShiftCReturnsNil() {
        XCTAssertNil(JunoHotkeyEventLine.commandCopyLine(
            keyCode: 8,
            modifierFlags: [.command, .shift],
            isRepeat: false
        ))
    }

    func testControlCReturnsNil() {
        XCTAssertNil(JunoHotkeyEventLine.commandCopyLine(
            keyCode: 8,
            modifierFlags: [.control],
            isRepeat: false
        ))
    }

    func testCopyReadyPolicyRequiresCopyLineIdleAndTrimmedTranscript() {
        XCTAssertTrue(JunoCopyReadyShortcutPolicy.shouldCopyReadyTranscript(
            hotkeyLine: "COPY",
            copyableTranscript: "  copied text\n",
            hudStateWire: "idle"
        ))

        XCTAssertFalse(JunoCopyReadyShortcutPolicy.shouldCopyReadyTranscript(
            hotkeyLine: "ESC",
            copyableTranscript: "copied text",
            hudStateWire: "idle"
        ))
        XCTAssertFalse(JunoCopyReadyShortcutPolicy.shouldCopyReadyTranscript(
            hotkeyLine: "COPY",
            copyableTranscript: "copied text",
            hudStateWire: "listening"
        ))
        XCTAssertFalse(JunoCopyReadyShortcutPolicy.shouldCopyReadyTranscript(
            hotkeyLine: "COPY",
            copyableTranscript: nil,
            hudStateWire: "idle"
        ))
        XCTAssertFalse(JunoCopyReadyShortcutPolicy.shouldCopyReadyTranscript(
            hotkeyLine: "COPY",
            copyableTranscript: " \n\t ",
            hudStateWire: "idle"
        ))
    }

    func testDictationShortcutSuppressionOnlyWhenCopyReady() {
        XCTAssertTrue(JunoCopyReadyShortcutPolicy.shouldSuppressDictationShortcut(
            copyableTranscript: "copied text",
            hudStateWire: "idle"
        ))

        XCTAssertFalse(JunoCopyReadyShortcutPolicy.shouldSuppressDictationShortcut(
            copyableTranscript: "copied text",
            hudStateWire: "listening"
        ))
        XCTAssertFalse(JunoCopyReadyShortcutPolicy.shouldSuppressDictationShortcut(
            copyableTranscript: nil,
            hudStateWire: "idle"
        ))
        XCTAssertFalse(JunoCopyReadyShortcutPolicy.shouldSuppressDictationShortcut(
            copyableTranscript: " \n\t ",
            hudStateWire: "idle"
        ))
    }
}
