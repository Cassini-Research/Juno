import XCTest
@testable import JunoShell

/// Regression coverage for issue #108: cancelling dictation persisted the HUD
/// reveal animation's drawn prefix instead of the engine's committed text.
///
/// All transcript text in this file is synthetic.
final class JunoHUDCancelDraftTests: XCTestCase {

    /// The reveal is character-granular for short text: 15 characters produce
    /// 15 one-character steps, minus the two that collapse onto the previous
    /// step once trailing whitespace is trimmed ("Ready " -> "Ready").
    func testCommittedRevealIsCharacterGranularForShortText() {
        let target = "Ready to start."
        let steps = HUDCommittedTextReveal.steps(current: "", target: target)

        XCTAssertEqual(steps.first, "R")
        XCTAssertEqual(steps.last, target)
        XCTAssertEqual(steps.count, 13)
        for step in steps {
            XCTAssertTrue(
                target.hasPrefix(step),
                "every reveal step must be a prefix of the target, got \(step)"
            )
        }
        // Any step before the last is a partial draw of already-final engine
        // text — several of them cut mid-word. That is what made the cancel
        // path lossy.
        XCTAssertTrue(steps.contains("Ready to sta"))
    }

    /// Longer text is revealed in multi-character chunks (96 steps max), so the
    /// step count stays bounded while granularity grows.
    func testCommittedRevealChunksLongTextIntoBoundedSteps() {
        let target = String(repeating: "ab ", count: 100).trimmingCharacters(in: .whitespaces)
        let steps = HUDCommittedTextReveal.steps(current: "", target: target)

        XCTAssertLessThanOrEqual(steps.count, 96)
        XCTAssertEqual(steps.last, target)
    }

    /// A cancel landing while the reveal is mid-flight must persist the full
    /// committed text the engine produced, not the prefix on screen.
    func testCancelDraftUsesEngineCommittedTextNotRevealPrefix() {
        let controller = DictationController()
        let committed = "Ready to start."

        // One engine preview chunk. The reveal draws its first step
        // synchronously and schedules the rest, so the HUD is now showing a
        // strict prefix of the engine's committed text.
        controller.ingestEnginePreviewChunk(committed: committed, tail: "")

        let drawn = controller.liveDisplayTranscript
        XCTAssertFalse(drawn.isEmpty, "the reveal should have drawn its first step")
        XCTAssertTrue(committed.hasPrefix(drawn))
        XCTAssertLessThan(
            drawn.count,
            committed.count,
            "precondition: the reveal must still be in flight for this test to mean anything"
        )

        XCTAssertEqual(controller.hudCancelDraftTranscript(), committed)
        // The HUD itself is settled too, so any later read agrees.
        XCTAssertEqual(controller.liveDisplayTranscript, committed)
    }

    /// Resolving is idempotent: with the reveal already settled there is
    /// nothing left to finish and the resolver reports the same text.
    func testCancelDraftIsIdempotentOnceTheRevealIsSettled() {
        let controller = DictationController()
        let committed = "Ready to start."

        controller.ingestEnginePreviewChunk(committed: committed, tail: "")
        XCTAssertEqual(controller.hudCancelDraftTranscript(), committed)
        XCTAssertEqual(controller.hudCancelDraftTranscript(), committed)
    }

    /// No engine text at all: the resolver still yields nothing, so the cancel
    /// path stays a no-op rather than writing an empty history row.
    func testCancelDraftIsEmptyWithoutEngineText() {
        let controller = DictationController()
        XCTAssertEqual(controller.hudCancelDraftTranscript(), "")
    }
}
