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
