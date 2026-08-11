import XCTest
@testable import JunoShell

final class JunoShellRuntimeTests: XCTestCase {
    func testStartHotkeyBridgeIfOnboardingCompletedSkipsBeforeCompletion() {
        let runtime = JunoShellRuntime()
        var startCount = 0
        runtime.startHotkeyBridge = {
            startCount += 1
        }

        runtime.startHotkeyBridgeIfOnboardingCompleted(onboardingCompleted: false)

        XCTAssertEqual(startCount, 0)
    }

    func testStartHotkeyBridgeIfOnboardingCompletedStartsAfterCompletion() {
        let runtime = JunoShellRuntime()
        var startCount = 0
        runtime.startHotkeyBridge = {
            startCount += 1
        }

        runtime.startHotkeyBridgeIfOnboardingCompleted(onboardingCompleted: true)

        XCTAssertEqual(startCount, 1)
    }
}
