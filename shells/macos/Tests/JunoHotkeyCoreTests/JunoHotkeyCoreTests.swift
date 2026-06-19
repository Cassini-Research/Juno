import AppKit
import CoreGraphics
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

    func testFnGlobeBarePressConsumesAndEmitsDown() {
        let outcome = JunoFnGlobeKeyPolicy.decide(flags: .maskSecondaryFn, fnWasHeld: false)
        XCTAssertEqual(outcome.decision.edge, .down)
        XCTAssertTrue(outcome.decision.consume)
        XCTAssertTrue(outcome.fnNowHeld)
        XCTAssertEqual(JunoFnGlobeKeyPolicy.stdoutLine(for: outcome.decision.edge), "FN_DOWN")
    }

    func testFnGlobeBareReleaseConsumesAndEmitsUp() {
        let outcome = JunoFnGlobeKeyPolicy.decide(flags: [], fnWasHeld: true)
        XCTAssertEqual(outcome.decision.edge, .up)
        XCTAssertTrue(outcome.decision.consume)
        XCTAssertFalse(outcome.fnNowHeld)
        XCTAssertEqual(JunoFnGlobeKeyPolicy.stdoutLine(for: outcome.decision.edge), "FN_UP")
    }

    func testFnGlobeWithCommandPassesThrough() {
        let outcome = JunoFnGlobeKeyPolicy.decide(
            flags: [.maskSecondaryFn, .maskCommand],
            fnWasHeld: false
        )
        XCTAssertEqual(outcome.decision, .none)
        XCTAssertFalse(outcome.fnNowHeld)
    }

    func testFnGlobeNoChangeIsNoOp() {
        let outcome = JunoFnGlobeKeyPolicy.decide(flags: .maskSecondaryFn, fnWasHeld: true)
        XCTAssertEqual(outcome.decision, .none)
        XCTAssertTrue(outcome.fnNowHeld)
    }

    func testHotkeyLaunchArgumentsIncludeConsumeFlagOnlyForFnMode() {
        XCTAssertEqual(
            JunoFnGlobeKeyPolicy.hotkeyLaunchArguments(consumeFn: true),
            [JunoFnGlobeKeyPolicy.consumeFnLaunchFlag]
        )
        XCTAssertTrue(JunoFnGlobeKeyPolicy.hotkeyLaunchArguments(consumeFn: false).isEmpty)
    }
}
