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

// MARK: - JunoSystemRequirements pure logic

final class JunoSystemRequirementsTests: XCTestCase {
    private func snapshot(os: Int, gb: Int, chip: String = "Apple M3") -> JunoSystemRequirements.Snapshot {
        JunoSystemRequirements.Snapshot(osMajorVersion: os, memoryGB: gb, chipName: chip)
    }

    func testHardOSFloorAndSoftMemoryFloor() {
        let ok = snapshot(os: 15, gb: 16)
        XCTAssertTrue(ok.meetsMinimumOS)
        XCTAssertTrue(ok.meetsMinimumMemory)
        XCTAssertTrue(ok.meetsAllRequirements)
        XCTAssertNil(ok.unsupportedOSMessage)
        XCTAssertNil(ok.onboardingWarningMessage)

        // OS below Sequoia → hard-block message, regardless of memory.
        let oldOS = snapshot(os: 14, gb: 64)
        XCTAssertFalse(oldOS.meetsMinimumOS)
        XCTAssertFalse(oldOS.meetsAllRequirements)
        XCTAssertNotNil(oldOS.unsupportedOSMessage)
        // OS-too-old is a hard block, never surfaced as a soft onboarding warning.
        XCTAssertNil(oldOS.onboardingWarningMessage)

        // Supported OS but below the memory floor → soft warning, no block.
        let lowRAM = snapshot(os: 15, gb: 8)
        XCTAssertTrue(lowRAM.meetsMinimumOS)
        XCTAssertFalse(lowRAM.meetsMinimumMemory)
        XCTAssertFalse(lowRAM.meetsAllRequirements)
        XCTAssertNil(lowRAM.unsupportedOSMessage)
        XCTAssertEqual(
            lowRAM.onboardingWarningMessage,
            "Juno runs best on Macs with at least 16 GB of memory. This Mac has 8 GB, so dictation and live preview may be slow."
        )
    }

    func testAppleChipGenerationParsing() {
        XCTAssertEqual(JunoSystemRequirements.appleChipGeneration(from: "Apple M1"), 1)
        XCTAssertEqual(JunoSystemRequirements.appleChipGeneration(from: "Apple M3 Pro"), 3)
        XCTAssertEqual(JunoSystemRequirements.appleChipGeneration(from: "Apple M4 Max"), 4)
        XCTAssertEqual(JunoSystemRequirements.appleChipGeneration(from: "apple m2 ultra"), 2) // case-insensitive
        XCTAssertNil(JunoSystemRequirements.appleChipGeneration(from: "Intel(R) Core(TM) i9-9980HK"))
        XCTAssertNil(JunoSystemRequirements.appleChipGeneration(from: ""))
        XCTAssertNil(JunoSystemRequirements.appleChipGeneration(from: "Apple M"))
    }
}

// MARK: - JunoPreviewEligibility pure logic
// (gates hudLiveTranscriptionsEnabled — preview follows the hard OS floor;
// memory is warning-only.)

final class JunoPreviewEligibilityTests: XCTestCase {
    private func snapshot(os: Int, gb: Int, chip: String = "Apple M3") -> JunoPreviewEligibility.Snapshot {
        JunoPreviewEligibility.Snapshot(osMajorVersion: os, memoryGB: gb, chipName: chip)
    }

    func testEligibilityRequiresSequoiaOnly() {
        XCTAssertTrue(snapshot(os: 15, gb: 16).isEligible)
        XCTAssertTrue(snapshot(os: 15, gb: 8).isEligible)    // memory is warning-only
        XCTAssertTrue(snapshot(os: 26, gb: 64).isEligible)
        XCTAssertFalse(snapshot(os: 14, gb: 64).isEligible)  // OS too old
    }

    func testUnavailableMessageForUnsupportedOS() {
        let low = snapshot(os: 15, gb: 8)
        XCTAssertNil(low.unavailableMessage)
        XCTAssertEqual(snapshot(os: 14, gb: 64).unavailableMessage, "Live preview isn’t available on this Mac.")
        XCTAssertNil(snapshot(os: 15, gb: 16).unavailableMessage)
    }

    func testWarningMessageNearMemoryFloor() {
        XCTAssertEqual(
            snapshot(os: 15, gb: 8).warningMessage,
            "Live preview can slow final transcription on 8 GB Macs."
        )
        XCTAssertEqual(
            snapshot(os: 15, gb: 16).warningMessage,
            "Live preview can slow final transcription on 16 GB Macs."
        )
        XCTAssertNil(snapshot(os: 15, gb: 64).warningMessage)  // ample memory
        XCTAssertNil(snapshot(os: 14, gb: 16).warningMessage)  // ineligible → no warning
    }
}
