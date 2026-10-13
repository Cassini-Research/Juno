import XCTest
@testable import JunoShell

/// History rows for takes the engine's guards rejected. Entries are decoded
/// from synthetic broker JSON so the test covers the same decode path the
/// History list uses.
final class JunoHistoryFailureLabelTests: XCTestCase {
    private func entry(failureReason: String) throws -> UtteranceHistoryEntry {
        let json = """
        {
          "utterance_id": "synthetic-utterance",
          "ts_unix_ms": 1785265200000,
          "transcript": "",
          "raw_transcript": "",
          "failure_reason": "\(failureReason)",
          "context": {"app_name": "Notes", "app_bundle_id": "com.apple.Notes"},
          "recovery": {
            "severity": "info",
            "category": "capture",
            "actions": [],
            "audio_present": false,
            "has_text": false,
            "failure_code": "\(failureReason)"
          }
        }
        """
        return try BrokerDecode.decoder.decode(UtteranceHistoryEntry.self, from: Data(json.utf8))
    }

    func testNoSpeechVerdictsReadLikeTheExistingEmptyAudioRow() throws {
        let emptyAudioLabel = try entry(failureReason: "empty_audio").displayFailureReason

        XCTAssertEqual(emptyAudioLabel, "Juno did not hear enough audio to transcribe.")
        for reason in ["low_signal_audio", "silence_hallucination"] {
            let row = try entry(failureReason: reason)

            XCTAssertEqual(row.displayFailureReason, emptyAudioLabel, reason)
            // With no transcript the label is also the row's primary line.
            XCTAssertEqual(row.historyPrimaryLine, emptyAudioLabel, reason)
        }
    }

    func testHallucinatedTranscriptRowNeitherClaimsSilenceNorShowsARawCode() throws {
        let row = try entry(failureReason: "hallucinated_transcript")

        XCTAssertEqual(
            row.displayFailureReason,
            "Juno discarded an unreliable transcript instead of inserting it."
        )
        XCTAssertNotEqual(
            row.displayFailureReason,
            try entry(failureReason: "silence_hallucination").displayFailureReason
        )
    }

    func testUnclassifiedReasonsKeepTheirExistingLabels() throws {
        XCTAssertEqual(
            try entry(failureReason: "transcribe_failed").displayFailureReason,
            "Transcribe Failed"
        )
        XCTAssertEqual(
            try entry(failureReason: "empty_final_text").displayFailureReason,
            "Dictation produced no text to insert."
        )
        XCTAssertEqual(
            try entry(failureReason: "paste_failed").displayFailureReason,
            "Juno could not insert text in Notes."
        )
    }
}
