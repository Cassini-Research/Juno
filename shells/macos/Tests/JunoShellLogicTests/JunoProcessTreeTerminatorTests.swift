import XCTest
@testable import JunoShell

final class JunoProcessTreeTerminatorTests: XCTestCase {
    func testDescendantsWalksNestedChildrenWithoutSiblings() {
        let rows = [
            JunoProcessTreeTerminator.ProcessRow(pid: 10, ppid: 1),
            JunoProcessTreeTerminator.ProcessRow(pid: 11, ppid: 10),
            JunoProcessTreeTerminator.ProcessRow(pid: 12, ppid: 10),
            JunoProcessTreeTerminator.ProcessRow(pid: 13, ppid: 11),
            JunoProcessTreeTerminator.ProcessRow(pid: 14, ppid: 99),
            JunoProcessTreeTerminator.ProcessRow(pid: 15, ppid: 14),
        ]

        XCTAssertEqual(
            JunoProcessTreeTerminator.descendants(of: 10, rows: rows),
            [11, 12, 13]
        )
    }
}
