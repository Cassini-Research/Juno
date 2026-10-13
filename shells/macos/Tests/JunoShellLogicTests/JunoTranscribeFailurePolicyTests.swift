import XCTest
@testable import JunoShell

final class JunoTranscribeFailurePolicyTests: XCTestCase {
    func testSilenceVerdictsShareTheNoSpeechOutcomeAndKeepTheEngineCode() {
        for code in ["silence_hallucination", "empty_audio"] {
            let disposition = JunoTranscribeFailurePolicy.disposition(forEngineErrorCode: code)

            XCTAssertEqual(disposition.kind, .noSpeechDetected, code)
            XCTAssertEqual(disposition.failureReason, code, code)
            XCTAssertNotEqual(disposition.failureReason, "transcribe_failed", code)
            XCTAssertEqual(disposition.hudStateWire, "error:low_signal_audio", code)
            XCTAssertEqual(disposition.speechHint, "No clear speech detected.", code)
            // The live-caption draft stays recoverable through the copy
            // fallback: a guard can misfire on quiet real speech.
            XCTAssertFalse(disposition.skipsLiveDraftFallback, code)
        }
    }

    func testLowSignalAudioKeepsItsExistingOutcome() {
        let disposition = JunoTranscribeFailurePolicy.disposition(forEngineErrorCode: "low_signal_audio")

        XCTAssertEqual(disposition.kind, .noSpeechDetected)
        XCTAssertEqual(disposition.failureReason, "low_signal_audio")
        XCTAssertEqual(disposition.hudStateWire, "error:low_signal_audio")
        XCTAssertEqual(disposition.speechHint, "No clear speech detected.")
        XCTAssertTrue(disposition.skipsLiveDraftFallback)
    }

    func testHallucinatedTranscriptKeepsItsCodeWithoutClaimingSilence() {
        let disposition = JunoTranscribeFailurePolicy.disposition(forEngineErrorCode: "hallucinated_transcript")

        XCTAssertEqual(disposition.kind, .transcriptRejected)
        XCTAssertEqual(disposition.failureReason, "hallucinated_transcript")
        // The user may have spoken: keep the existing "couldn't transcribe"
        // HUD state and never surface a "no speech" hint for it.
        XCTAssertEqual(disposition.hudStateWire, "error:transcribe_failed")
        XCTAssertNil(disposition.speechHint)
        XCTAssertFalse(disposition.skipsLiveDraftFallback)
    }

    func testRealEngineAndTransportFailuresStayTranscribeFailed() {
        let engineCodes: [String?] = [
            "transcribe_failed",
            "inference_cancelled",
            "engine_incompatible",
            "capability_blocked",
            "a_code_this_shell_has_never_seen",
            "",
            nil,
        ]
        for code in engineCodes {
            let disposition = JunoTranscribeFailurePolicy.disposition(forEngineErrorCode: code)
            let label = code ?? "nil"

            XCTAssertEqual(disposition.kind, .engineFailure, label)
            XCTAssertEqual(disposition.failureReason, "transcribe_failed", label)
            XCTAssertEqual(disposition.hudStateWire, "error:transcribe_failed", label)
            XCTAssertNil(disposition.speechHint, label)
            XCTAssertFalse(disposition.skipsLiveDraftFallback, label)
        }
    }

    func testEngineCodeIsMatchedAfterTrimmingWhitespace() {
        let disposition = JunoTranscribeFailurePolicy.disposition(forEngineErrorCode: " silence_hallucination\n")

        XCTAssertEqual(disposition.failureReason, "silence_hallucination")
    }

    /// The synthetic reproduction from the bug report: the engine's response
    /// for a near-silent take, fed through the parser the shell really uses.
    func testEngineSilenceRejectionResponseIsNotSurfacedAsTranscribeFailed() throws {
        let response: [String: Any] = [
            "ok": false,
            "error": "silence_hallucination",
            "error_code": "silence_hallucination",
        ]

        guard case .failure(let error) = JunoBroker.parseTranscribeResponse(response) else {
            return XCTFail("an ok:false engine response must parse as a failure")
        }
        let disposition = JunoTranscribeFailurePolicy.disposition(forBrokerError: error)

        XCTAssertEqual(disposition.kind, .noSpeechDetected)
        XCTAssertEqual(disposition.failureReason, "silence_hallucination")
        XCTAssertEqual(disposition.hudStateWire, "error:low_signal_audio")
    }

    func testEngineErrorWithoutACodeStaysTranscribeFailed() {
        let response: [String: Any] = ["ok": false, "error": "transcribe_failed: decoder crashed"]

        guard case .failure(let error) = JunoBroker.parseTranscribeResponse(response) else {
            return XCTFail("an ok:false engine response must parse as a failure")
        }

        XCTAssertEqual(
            JunoTranscribeFailurePolicy.disposition(forBrokerError: error).failureReason,
            "transcribe_failed"
        )
    }

    func testTransportErrorStaysTranscribeFailed() {
        let transport = NSError(domain: NSURLErrorDomain, code: NSURLErrorCannotConnectToHost)

        let disposition = JunoTranscribeFailurePolicy.disposition(forBrokerError: transport)

        XCTAssertEqual(disposition.kind, .engineFailure)
        XCTAssertEqual(disposition.failureReason, "transcribe_failed")
    }

    func testNoSpeechStateDecodesToATypedReasonAndRoundTrips() {
        let state = HUDState.from(wireString: "error:low_signal_audio")

        XCTAssertEqual(state, .error(reason: .lowSignalAudio))
        XCTAssertEqual(state.wireString, "error:low_signal_audio")
        XCTAssertTrue(state.isErrorOrBlocked)
    }

    func testNoSpeechStateShowsNoSpeechCopyInsteadOfAGenericFailure() throws {
        let copy = try XCTUnwrap(
            JunoTranscribeFailurePolicy.hudErrorCopy(for: HUDState.from(wireString: "error:low_signal_audio"))
        )

        XCTAssertEqual(copy.title, "No clear speech detected")
        XCTAssertEqual(copy.message, "Juno did not hear enough audio to transcribe.")
    }

    func testOtherErrorStatesKeepTheIslandsOwnCopy() {
        let states = [
            "error:transcribe_failed",
            "error:mic_no_audio",
            "error:some other failure",
            "blocked:ax_permission_missing",
            "refining",
            "idle",
        ]
        for wire in states {
            XCTAssertNil(
                JunoTranscribeFailurePolicy.hudErrorCopy(for: HUDState.from(wireString: wire)),
                wire
            )
        }
    }

    func testHistoryLabelsCoverGuardRejectionsOnly() {
        for reason in ["low_signal_audio", "silence_hallucination", "empty_audio"] {
            XCTAssertEqual(
                JunoTranscribeFailurePolicy.historyLabel(forFailureReason: reason),
                "Juno did not hear enough audio to transcribe.",
                reason
            )
        }
        XCTAssertEqual(
            JunoTranscribeFailurePolicy.historyLabel(forFailureReason: "hallucinated_transcript"),
            "Juno discarded an unreliable transcript instead of inserting it."
        )
        for reason in ["transcribe_failed", "paste_failed", "user_cancelled_hud", "empty_final_text"] {
            XCTAssertNil(JunoTranscribeFailurePolicy.historyLabel(forFailureReason: reason), reason)
        }
    }
}
