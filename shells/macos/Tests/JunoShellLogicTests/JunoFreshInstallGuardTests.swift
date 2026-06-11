// JunoFreshInstallGuardTests.swift
//
// JunoFreshInstallGuard.runOnce() is testable here because its permission
// reads are non-prompting: AVCaptureDevice.authorizationStatus(for:) and
// AXIsProcessTrusted() only query current TCC state (they never present a
// system dialog — that requires requestAccess / AXIsProcessTrustedWithOptions
// with the prompt option).
//
// Note: with currentOnboardingRequirementsVersion == 1, the
// "completedRequirements < currentRequirements" reset branch is unreachable
// (any stored version <= 0 is caught by the predate-marker branch first),
// so only the reachable branches are exercised.

import XCTest
@testable import JunoShell

final class JunoFreshInstallGuardTests: JunoDefaultsRestoringTestCase {
    func testNotOnboardedIsANoOp() {
        UserDefaults.standard.set(false, forKey: JunoUserDefaults.onboardingCompletedKey)
        JunoFreshInstallGuard.runOnce()
        XCTAssertFalse(JunoUserDefaults.onboardingCompleted)
        // Version marker must not be stamped while still in onboarding.
        XCTAssertNil(UserDefaults.standard.object(forKey: JunoUserDefaults.onboardingRequirementsVersionKey))
    }

    func testInstallPredatingVersionMarkerIsGrandfatheredIn() {
        // Onboarded install from before the requirements marker existed:
        // completed flag set directly, no version key.
        UserDefaults.standard.set(true, forKey: JunoUserDefaults.onboardingCompletedKey)
        UserDefaults.standard.removeObject(forKey: JunoUserDefaults.onboardingRequirementsVersionKey)

        JunoFreshInstallGuard.runOnce()

        // Onboarding stays completed; version is stamped to current.
        XCTAssertTrue(JunoUserDefaults.onboardingCompleted)
        XCTAssertEqual(
            JunoUserDefaults.onboardingRequirementsVersion,
            JunoUserDefaults.currentOnboardingRequirementsVersion
        )
    }

    func testSteadyStateIsIdempotent() {
        JunoUserDefaults.onboardingCompleted = true // stamps current version
        JunoUserDefaults.incrementDictationCompletedCount()

        JunoFreshInstallGuard.runOnce()
        JunoFreshInstallGuard.runOnce()

        XCTAssertTrue(JunoUserDefaults.onboardingCompleted)
        XCTAssertEqual(
            JunoUserDefaults.onboardingRequirementsVersion,
            JunoUserDefaults.currentOnboardingRequirementsVersion
        )
        // Steady state must not clear user progress.
        XCTAssertEqual(JunoUserDefaults.dictationCompletedCount, 1)
    }

    func testFutureVersionStampIsLeftAlone() {
        // A downgrade scenario: stored version is newer than this build's.
        UserDefaults.standard.set(true, forKey: JunoUserDefaults.onboardingCompletedKey)
        JunoUserDefaults.onboardingRequirementsVersion =
            JunoUserDefaults.currentOnboardingRequirementsVersion + 5

        JunoFreshInstallGuard.runOnce()

        XCTAssertTrue(JunoUserDefaults.onboardingCompleted)
        XCTAssertEqual(
            JunoUserDefaults.onboardingRequirementsVersion,
            JunoUserDefaults.currentOnboardingRequirementsVersion + 5
        )
    }
}

// MARK: - JunoPreviewEligibility pure logic
// (gates hudLiveTranscriptionsEnabled, tested above)

final class JunoPreviewEligibilityTests: XCTestCase {
    func testAppleChipGenerationParsing() {
        XCTAssertEqual(JunoPreviewEligibility.appleChipGeneration(from: "Apple M1"), 1)
        XCTAssertEqual(JunoPreviewEligibility.appleChipGeneration(from: "Apple M3 Pro"), 3)
        XCTAssertEqual(JunoPreviewEligibility.appleChipGeneration(from: "Apple M4 Max"), 4)
        XCTAssertEqual(JunoPreviewEligibility.appleChipGeneration(from: "apple m2 ultra"), 2) // case-insensitive
        XCTAssertNil(JunoPreviewEligibility.appleChipGeneration(from: "Intel(R) Core(TM) i9-9980HK"))
        XCTAssertNil(JunoPreviewEligibility.appleChipGeneration(from: ""))
        XCTAssertNil(JunoPreviewEligibility.appleChipGeneration(from: "Apple M"))
    }

    func testEligibilityRequiresM3OrNewerAnd32GB() {
        func snapshot(_ gen: Int?, _ gb: Int) -> JunoPreviewEligibility.Snapshot {
            JunoPreviewEligibility.Snapshot(chipName: "Test", chipGeneration: gen, memoryGB: gb)
        }
        XCTAssertTrue(snapshot(3, 32).isEligible)
        XCTAssertTrue(snapshot(4, 128).isEligible)
        XCTAssertFalse(snapshot(2, 64).isEligible)   // chip too old
        XCTAssertFalse(snapshot(3, 16).isEligible)   // not enough memory
        XCTAssertFalse(snapshot(nil, 64).isEligible) // non-Apple-Silicon
    }

    func testUnavailableMessages() {
        let intel = JunoPreviewEligibility.Snapshot(chipName: "Intel", chipGeneration: nil, memoryGB: 64)
        XCTAssertEqual(intel.unavailableMessage, "Live preview requires Apple Silicon M3 or newer.")

        let m2 = JunoPreviewEligibility.Snapshot(chipName: "Apple M2", chipGeneration: 2, memoryGB: 64)
        XCTAssertEqual(
            m2.unavailableMessage,
            "Live preview requires Apple Silicon M3 or newer. This Mac reports Apple M2."
        )

        let lowMemory = JunoPreviewEligibility.Snapshot(chipName: "Apple M3", chipGeneration: 3, memoryGB: 16)
        XCTAssertEqual(
            lowMemory.unavailableMessage,
            "Live preview requires at least 32 GB memory. This Mac has 16 GB."
        )

        let eligible = JunoPreviewEligibility.Snapshot(chipName: "Apple M3", chipGeneration: 3, memoryGB: 32)
        XCTAssertNil(eligible.unavailableMessage)
    }

    func testWarningMessageOnlyForEligibleMacsUpTo64GB() {
        let small = JunoPreviewEligibility.Snapshot(chipName: "Apple M3", chipGeneration: 3, memoryGB: 32)
        XCTAssertEqual(small.warningMessage, "Live preview can hamper transcription performance on 32 GB Macs.")

        let big = JunoPreviewEligibility.Snapshot(chipName: "Apple M3", chipGeneration: 3, memoryGB: 128)
        XCTAssertNil(big.warningMessage)

        let ineligible = JunoPreviewEligibility.Snapshot(chipName: "Apple M2", chipGeneration: 2, memoryGB: 32)
        XCTAssertNil(ineligible.warningMessage)
    }
}
