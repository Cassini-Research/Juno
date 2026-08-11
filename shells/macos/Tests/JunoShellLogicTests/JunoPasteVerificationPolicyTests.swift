import XCTest
@testable import JunoShell

final class JunoPasteVerificationPolicyTests: XCTestCase {
    func testTerminalXTextAreaIsNotReliablePasteReadback() {
        XCTAssertFalse(
            JunoPasteVerificationPolicy.isReadbackReliable(
                bundleId: "com.terminalx",
                role: "AXTextArea",
                hasStringValue: true
            )
        )
    }

    func testNativeTextAreaWithValueRemainsReliablePasteReadback() {
        XCTAssertTrue(
            JunoPasteVerificationPolicy.isReadbackReliable(
                bundleId: "com.apple.TextEdit",
                role: "AXTextArea",
                hasStringValue: true
            )
        )
    }

    func testEmptyMonitorSnapshotCannotOverrideAcceptedPaste() {
        XCTAssertFalse(
            JunoPasteVerificationPolicy.shouldOfferCopyFallback(
                pasteWasAccepted: true,
                postPasteSnapshot: ""
            )
        )
    }
}
