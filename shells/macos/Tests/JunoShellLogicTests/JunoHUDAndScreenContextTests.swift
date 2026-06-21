import XCTest
@testable import JunoShell

final class JunoHUDAndScreenContextTests: XCTestCase {
    func testHUDRawTextKeepsSpokenParagraphCueOutOfEngineHint() {
        let store = HUDTranscriptStore()

        _ = store.applyPreviewRevision(
            committed: "Hello new paragraph next point",
            tail: ""
        )

        XCTAssertEqual(store.rawText, "Hello new paragraph next point")
        XCTAssertTrue(store.text.contains("\n\n"))
        XCTAssertTrue(store.text.contains("new paragraph"))
        XCTAssertFalse(store.rawText.contains("\n"))
    }

    func testSpokenBreakCueResolutionStripsLiteralWordsForCopyFallback() {
        // The copy/paste fallback must convert spoken cues to real breaks and
        // drop the literal words — never leak "new line"/"new paragraph" text.
        let line = HUDTranscriptStore.transcriptWithSpokenBreakCuesResolved(
            "first point new line second point"
        )
        XCTAssertEqual(line, "first point \nsecond point")
        XCTAssertFalse(line.contains("new line"))

        let para = HUDTranscriptStore.transcriptWithSpokenBreakCuesResolved(
            "intro new paragraph body"
        )
        XCTAssertEqual(para, "intro \n\nbody")
        XCTAssertFalse(para.contains("new paragraph"))

        // Determiner guard: genuine prose stays literal (no spurious break).
        let literal = HUDTranscriptStore.transcriptWithSpokenBreakCuesResolved(
            "the new line is short"
        )
        XCTAssertEqual(literal, "the new line is short")
    }

    func testCommittedShrinkAcceptsUnambiguousPunctuationCueSuffix() {
        let store = HUDTranscriptStore()
        _ = store.applyPreviewRevision(committed: "hello world comma", tail: "")
        XCTAssertEqual(store.rawText, "hello world comma")

        // A re-anchor that drops a trailing spoken punctuation cue ("comma")
        // is allowed to shrink the committed HUD.
        let changed = store.applyPreviewRevision(committed: "hello world", tail: "")
        XCTAssertTrue(changed)
        XCTAssertEqual(store.rawText, "hello world")
    }

    func testCommittedShrinkRefusesDeterminerProtectedCuePhrase() {
        let store = HUDTranscriptStore()
        _ = store.applyPreviewRevision(committed: "the new line", tail: "")
        XCTAssertEqual(store.rawText, "the new line")

        // "the new line" is real dictated prose (determiner guard), not a
        // spoken cue — a shorter re-anchor must NOT shrink those words away.
        let changed = store.applyPreviewRevision(committed: "the", tail: "")
        XCTAssertFalse(changed)
        XCTAssertEqual(store.rawText, "the new line")
    }

    func testScreenTermHarvesterDropsOCRJunkBeforeBiasing() {
        let terms = JunoScreenTermHarvester.distillTerms(from: [
            "onboardin9 Acces5ibility Rerninders rn0 coM JuDo NOvaD SettiThJs CityXyoTer bhlS.py atlons/Junty.app",
            "HUD NovaDesk OpenAI Cassini Research juno_v2.swift",
        ])

        for junk in [
            "onboardin9", "Acces5ibility", "Rerninders", "rn0", "coM",
            "JuDo", "NOvaD", "SettiThJs", "CityXyoTer", "bhlS.py",
            "atlons/Junty.app",
        ] {
            XCTAssertFalse(terms.contains(junk), "unexpected OCR junk term: \(junk)")
        }
        XCTAssertTrue(terms.contains("HUD"))
        XCTAssertTrue(terms.contains("NovaDesk"))
        XCTAssertTrue(terms.contains("OpenAI"))
        XCTAssertTrue(terms.contains("Cassini Research"))
        XCTAssertTrue(terms.contains("juno_v2.swift"))
    }
}
