import XCTest
@testable import JunoShell

final class JunoMemoryListRowsTests: XCTestCase {
    func testRowIdsFollowSemanticKeysAcrossSearchFiltering() {
        let entries: [[String: Any]] = [
            ["trigger": "alpha", "scope": "global"],
            ["trigger": "beta", "scope": "global"],
            ["trigger": "gamma", "scope": "email"],
        ]

        let rows = MemoryListRows.make(entries: entries, keyForEntry: snippetKey)
        XCTAssertEqual(rows.map(\.id), [
            "snippet:alpha:global",
            "snippet:beta:global",
            "snippet:gamma:email",
        ])

        let filtered = entries.filter {
            ($0["trigger"] as? String)?.contains("beta") ?? false
        }
        XCTAssertEqual(
            MemoryListRows.make(entries: filtered, keyForEntry: snippetKey).map(\.id),
            ["snippet:beta:global"]
        )
    }

    func testDuplicateSemanticKeysReceiveUniqueRowIds() {
        let entries: [[String: Any]] = [
            ["trigger": "signoff", "scope": "global"],
            ["trigger": "signoff", "scope": "global"],
            ["trigger": "signoff", "scope": "email"],
        ]

        let rows = MemoryListRows.make(entries: entries, keyForEntry: snippetKey)

        XCTAssertEqual(rows.map(\.key), [
            "snippet:signoff:global",
            "snippet:signoff:global",
            "snippet:signoff:email",
        ])
        XCTAssertEqual(rows.map(\.id), [
            "snippet:signoff:global",
            "snippet:signoff:global#2",
            "snippet:signoff:email",
        ])
        XCTAssertEqual(Set(rows.map(\.id)).count, rows.count)
    }

    private func snippetKey(_ entry: [String: Any]) -> String {
        let trigger = entry["trigger"] as? String ?? ""
        let scope = entry["scope"] as? String ?? "global"
        return "snippet:\(trigger):\(scope)"
    }
}
