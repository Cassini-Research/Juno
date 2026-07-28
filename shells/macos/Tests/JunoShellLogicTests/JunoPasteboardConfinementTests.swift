import AppKit
import XCTest
@testable import JunoShell

/// Regression coverage for the 1.0.9 (23) production crash:
/// NSRangeException in -[NSPasteboard _updateTypeCacheIfNeeded] when
/// SurfaceEditingModel's utility-queue capability poll read
/// NSPasteboard.general concurrently with main-thread pasteboard writes.
/// NSPasteboard's internal type cache is rebuilt without locking, so all
/// reads must be confined to the main thread — the thread every in-process
/// write already uses. These tests use a private named pasteboard so they
/// never touch the user's real clipboard.
final class JunoPasteboardConfinementTests: XCTestCase {
    private func makePasteboard() -> NSPasteboard {
        NSPasteboard(name: NSPasteboard.Name("com.juno.tests.pasteboard.\(UUID().uuidString)"))
    }

    func testConfinedReadFromBackgroundThreadSeesMainThreadWrite() {
        let pasteboard = makePasteboard()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        pasteboard.setString("juno-confined-read", forType: .string)

        var result: String?
        let done = expectation(description: "background read completed")
        DispatchQueue.global(qos: .utility).async {
            result = JunoLocalCapability.pasteboardStringConfinedToMain(pasteboard)
            done.fulfill()
        }
        // wait(for:) pumps the main run loop, which drains the main-queue
        // hop inside pasteboardStringConfinedToMain.
        wait(for: [done], timeout: 5)
        XCTAssertEqual(result, "juno-confined-read")
    }

    func testConfinedReadOnMainThreadDoesNotDeadlock() {
        let pasteboard = makePasteboard()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        pasteboard.setString("juno-main-read", forType: .string)
        XCTAssertEqual(
            JunoLocalCapability.pasteboardStringConfinedToMain(pasteboard),
            "juno-main-read"
        )
    }

    /// Crash canary: concurrent background reads through the confined helper
    /// while the main thread rewrites the pasteboard. With unconfined reads
    /// this access pattern aborts the process within seconds (verified
    /// against the production crash's stack); through the helper it must
    /// survive at full pressure.
    func testConcurrentConfinedReadsSurviveMainThreadWrites() {
        let pasteboard = makePasteboard()
        defer { pasteboard.releaseGlobally() }

        let deadline = Date().addingTimeInterval(1.5)
        let readerCount = 4
        let readersDone = expectation(description: "readers finished")
        readersDone.expectedFulfillmentCount = readerCount
        for index in 0..<readerCount {
            DispatchQueue.global(qos: index % 2 == 0 ? .utility : .default).async {
                while Date() < deadline {
                    _ = JunoLocalCapability.pasteboardStringConfinedToMain(pasteboard)
                }
                readersDone.fulfill()
            }
        }

        var writes = 0
        while Date() < deadline {
            pasteboard.clearContents()
            pasteboard.setString("juno-stress-\(writes)", forType: .string)
            writes += 1
            // Drain the readers' main-queue hops between writes.
            RunLoop.main.run(until: Date().addingTimeInterval(0.001))
        }
        wait(for: [readersDone], timeout: 10)
        XCTAssertGreaterThan(writes, 0)
    }
}
