import XCTest
@testable import JunoShell

final class JunoLocalCapabilityTests: XCTestCase {
    func testTerminalAppsAllowPasteFallbackForContainerFocus() {
        for bundleId in [
            "com.apple.Terminal",
            "com.googlecode.iterm2",
            "com.mitchellh.ghostty",
            "dev.warp.Warp-Stable",
        ] {
            XCTAssertTrue(
                JunoLocalCapability.knownPasteCentricAppAllowsFallback(
                    bundleId: bundleId,
                    role: "AXGroup"
                ),
                "expected \(bundleId) to allow paste fallback"
            )
        }
    }

    func testPasteFallbackStillRequiresSpecificFocusedUi() {
        XCTAssertFalse(
            JunoLocalCapability.knownPasteCentricAppAllowsFallback(
                bundleId: "com.apple.Terminal",
                role: "AXWindow"
            )
        )
        XCTAssertFalse(
            JunoLocalCapability.knownPasteCentricAppAllowsFallback(
                bundleId: "com.apple.Terminal",
                role: "AXApplication"
            )
        )
    }

    func testUnknownAppsDoNotAllowPasteFallback() {
        XCTAssertFalse(
            JunoLocalCapability.knownPasteCentricAppAllowsFallback(
                bundleId: "com.apple.systempreferences",
                role: "AXGroup"
            )
        )
    }
}
