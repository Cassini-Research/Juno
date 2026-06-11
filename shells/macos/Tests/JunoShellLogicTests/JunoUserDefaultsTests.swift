// JunoUserDefaultsTests.swift
//
// Pure-logic tests for JunoUserDefaults: clamping, whitelist validation,
// trimming/meaningful-character rules, legacy raw-value migration, boolean
// defaults, and counters.
//
// UserDefaults is global state for the test-host process, so every test
// snapshots the keys it touches in setUp and restores them in tearDown.

import XCTest
@testable import JunoShell

/// Base case that snapshots and restores every Juno defaults key the tests
/// touch, so `swift test` never leaves residue in the developer's defaults.
class JunoDefaultsRestoringTestCase: XCTestCase {
    private var snapshot: [String: Any?] = [:]

    /// Every key any test in this file may write.
    var trackedKeys: [String] {
        var keys = [
            JunoUserDefaults.onboardingCompletedKey,
            JunoUserDefaults.onboardingRequirementsVersionKey,
            JunoUserDefaults.preferredDisplayNameKey,
            JunoUserDefaults.onboardingBrandDelightShownKey,
            JunoUserDefaults.hudDelightAnimationsEnabledKey,
            JunoUserDefaults.hudDelightSoundEnabledKey,
            JunoUserDefaults.pauseSensitivitySecondsKey,
            JunoUserDefaults.hudPositionKey,
            JunoUserDefaults.hudLiveTranscriptionsEnabledKey,
            JunoUserDefaults.liveAdjudicationEnabledKey,
            JunoUserDefaults.whisperPreviewDefaultsMigratedKey,
            JunoUserDefaults.showInDockKey,
            JunoUserDefaults.micVoiceProcessingEnabledKey,
            JunoUserDefaults.hudShowDoneRowEnabledKey,
            JunoUserDefaults.languageModeKey,
            JunoUserDefaults.developerModeEnabledKey,
            JunoUserDefaults.saveLogsToFileEnabledKey,
            JunoUserDefaults.appearancePreferenceKey,
            JunoUserDefaults.actionsEnabledKey,
            JunoUserDefaults.actionsNotesSignatureEnabledKey,
            JunoUserDefaults.actionsOnboardingDecisionMadeKey,
            JunoUserDefaults.dictationCompletedCountKey,
            JunoUserDefaults.actionsNudgeShownKey,
            JunoUserDefaults.hudOpenSoundEnabledKey,
            "JunoRemindersNudgeColdDismissedAt",
        ]
        keys += JunoActionKind.allCases.map { "JunoActionsNudgeDismissedAt_\($0.rawValue)" }
        return keys
    }

    override func setUp() {
        super.setUp()
        let ud = UserDefaults.standard
        snapshot = [:]
        for key in trackedKeys {
            snapshot[key] = ud.object(forKey: key)
            ud.removeObject(forKey: key)
        }
    }

    override func tearDown() {
        let ud = UserDefaults.standard
        for key in trackedKeys {
            if let value = snapshot[key] ?? nil {
                ud.set(value, forKey: key)
            } else {
                ud.removeObject(forKey: key)
            }
        }
        snapshot = [:]
        super.tearDown()
    }
}

// MARK: - pauseSensitivitySeconds

final class PauseSensitivityTests: JunoDefaultsRestoringTestCase {
    func testDefaultWhenUnsetIs1Point4() {
        XCTAssertEqual(JunoUserDefaults.pauseSensitivitySeconds, 1.4, accuracy: 0.0001)
    }

    func testSetterClampsBelowLowerBound() {
        JunoUserDefaults.pauseSensitivitySeconds = 0.1
        XCTAssertEqual(JunoUserDefaults.pauseSensitivitySeconds, 0.8, accuracy: 0.0001)
        // The clamped value (not the raw input) is what gets persisted.
        XCTAssertEqual(
            UserDefaults.standard.double(forKey: JunoUserDefaults.pauseSensitivitySecondsKey),
            0.8,
            accuracy: 0.0001
        )
    }

    func testSetterClampsAboveUpperBound() {
        JunoUserDefaults.pauseSensitivitySeconds = 99.0
        XCTAssertEqual(JunoUserDefaults.pauseSensitivitySeconds, 3.0, accuracy: 0.0001)
    }

    func testInRangeValueRoundTrips() {
        JunoUserDefaults.pauseSensitivitySeconds = 2.2
        XCTAssertEqual(JunoUserDefaults.pauseSensitivitySeconds, 2.2, accuracy: 0.0001)
    }

    func testBoundaryValuesAreKept() {
        JunoUserDefaults.pauseSensitivitySeconds = 0.8
        XCTAssertEqual(JunoUserDefaults.pauseSensitivitySeconds, 0.8, accuracy: 0.0001)
        JunoUserDefaults.pauseSensitivitySeconds = 3.0
        XCTAssertEqual(JunoUserDefaults.pauseSensitivitySeconds, 3.0, accuracy: 0.0001)
    }

    func testGetterClampsOutOfRangeStoredValue() {
        // Simulate a stale/hand-edited plist value bypassing the setter.
        UserDefaults.standard.set(0.05, forKey: JunoUserDefaults.pauseSensitivitySecondsKey)
        XCTAssertEqual(JunoUserDefaults.pauseSensitivitySeconds, 0.8, accuracy: 0.0001)
        UserDefaults.standard.set(42.0, forKey: JunoUserDefaults.pauseSensitivitySecondsKey)
        XCTAssertEqual(JunoUserDefaults.pauseSensitivitySeconds, 3.0, accuracy: 0.0001)
    }
}

// MARK: - languageMode

final class LanguageModeTests: JunoDefaultsRestoringTestCase {
    func testDefaultIsAuto() {
        XCTAssertEqual(JunoUserDefaults.languageMode, "auto")
    }

    func testAllWhitelistedValuesRoundTrip() {
        for mode in ["auto", "en", "pair:en,hi", "zh", "es", "keep_original"] {
            JunoUserDefaults.languageMode = mode
            XCTAssertEqual(JunoUserDefaults.languageMode, mode)
        }
    }

    func testInvalidStoredValueFallsBackToAuto() {
        // The setter does not validate; the getter is the gate.
        JunoUserDefaults.languageMode = "klingon"
        XCTAssertEqual(JunoUserDefaults.languageMode, "auto")
        UserDefaults.standard.set("fr", forKey: JunoUserDefaults.languageModeKey)
        XCTAssertEqual(JunoUserDefaults.languageMode, "auto")
    }

    func testNonStringStoredValueFallsBackToAuto() {
        UserDefaults.standard.set(true, forKey: JunoUserDefaults.languageModeKey)
        // bool(true) is bridged to "1" by string(forKey:), which is not whitelisted.
        XCTAssertEqual(JunoUserDefaults.languageMode, "auto")
    }
}

// MARK: - preferredDisplayName + meaningfulNameCharacterCount

final class PreferredDisplayNameTests: JunoDefaultsRestoringTestCase {
    func testNilWhenUnset() {
        XCTAssertNil(JunoUserDefaults.preferredDisplayName)
    }

    func testSetterTrimsWhitespaceAndNewlines() {
        JunoUserDefaults.preferredDisplayName = "  Paresh \n"
        XCTAssertEqual(JunoUserDefaults.preferredDisplayName, "Paresh")
        XCTAssertEqual(
            UserDefaults.standard.string(forKey: JunoUserDefaults.preferredDisplayNameKey),
            "Paresh"
        )
    }

    func testSettingNilOrWhitespaceRemovesKey() {
        JunoUserDefaults.preferredDisplayName = "Paresh"
        JunoUserDefaults.preferredDisplayName = nil
        XCTAssertNil(UserDefaults.standard.object(forKey: JunoUserDefaults.preferredDisplayNameKey))

        JunoUserDefaults.preferredDisplayName = "Paresh"
        JunoUserDefaults.preferredDisplayName = "   \n "
        XCTAssertNil(UserDefaults.standard.object(forKey: JunoUserDefaults.preferredDisplayNameKey))
    }

    func testNamesWithFewerThanThreeAlphanumericsAreRejectedBySetter() {
        JunoUserDefaults.preferredDisplayName = "ab"
        XCTAssertNil(JunoUserDefaults.preferredDisplayName)
        XCTAssertNil(UserDefaults.standard.object(forKey: JunoUserDefaults.preferredDisplayNameKey))

        JunoUserDefaults.preferredDisplayName = "a-b!"
        XCTAssertNil(JunoUserDefaults.preferredDisplayName)
    }

    func testThreeAlphanumericsIsEnough() {
        JunoUserDefaults.preferredDisplayName = "Al1"
        XCTAssertEqual(JunoUserDefaults.preferredDisplayName, "Al1")
    }

    func testGetterReturnsNilForLowMeaningfulCountStoredRaw() {
        // A short value written directly (legacy build, hand-edited plist)
        // is filtered out on read even though it is stored.
        UserDefaults.standard.set(" ab ", forKey: JunoUserDefaults.preferredDisplayNameKey)
        XCTAssertNil(JunoUserDefaults.preferredDisplayName)
    }

    func testPunctuationOnlyNameIsNil() {
        UserDefaults.standard.set("!!!", forKey: JunoUserDefaults.preferredDisplayNameKey)
        XCTAssertNil(JunoUserDefaults.preferredDisplayName)
    }

    func testMeaningfulNameCharacterCount() {
        XCTAssertEqual(JunoUserDefaults.meaningfulNameCharacterCount(""), 0)
        XCTAssertEqual(JunoUserDefaults.meaningfulNameCharacterCount("abc"), 3)
        XCTAssertEqual(JunoUserDefaults.meaningfulNameCharacterCount("a-b"), 2)
        XCTAssertEqual(JunoUserDefaults.meaningfulNameCharacterCount("a1!b2?"), 4)
        XCTAssertEqual(JunoUserDefaults.meaningfulNameCharacterCount("... --- ..."), 0)
        // Emoji are not alphanumerics.
        XCTAssertEqual(JunoUserDefaults.meaningfulNameCharacterCount("😀😀😀"), 0)
        // Non-Latin letters count (CharacterSet.alphanumerics is Unicode-aware).
        XCTAssertEqual(JunoUserDefaults.meaningfulNameCharacterCount("张伟明"), 3)
    }
}

// MARK: - HUDPosition

final class HUDPositionTests: JunoDefaultsRestoringTestCase {
    func testEnumCasesAndRawValues() {
        XCTAssertEqual(JunoUserDefaults.HUDPosition.allCases, [.topCenter, .bottomCenter])
        XCTAssertEqual(JunoUserDefaults.HUDPosition.topCenter.rawValue, "top_center")
        XCTAssertEqual(JunoUserDefaults.HUDPosition.bottomCenter.rawValue, "bottom_center")
        for position in JunoUserDefaults.HUDPosition.allCases {
            XCTAssertEqual(position.id, position.rawValue)
        }
    }

    func testEnumTitles() {
        XCTAssertEqual(JunoUserDefaults.HUDPosition.topCenter.title, "Top center")
        XCTAssertEqual(JunoUserDefaults.HUDPosition.bottomCenter.title, "Bottom center")
    }

    func testDefaultIsTopCenter() {
        XCTAssertEqual(JunoUserDefaults.hudPosition, .topCenter)
    }

    func testRoundTrip() {
        JunoUserDefaults.hudPosition = .bottomCenter
        XCTAssertEqual(JunoUserDefaults.hudPosition, .bottomCenter)
        JunoUserDefaults.hudPosition = .topCenter
        XCTAssertEqual(JunoUserDefaults.hudPosition, .topCenter)
    }

    func testLegacyCursorFollowMigratesToTopCenter() {
        UserDefaults.standard.set("cursor_follow", forKey: JunoUserDefaults.hudPositionKey)
        XCTAssertEqual(JunoUserDefaults.hudPosition, .topCenter)
    }

    func testUnknownRawValueFallsBackToTopCenter() {
        UserDefaults.standard.set("left_edge", forKey: JunoUserDefaults.hudPositionKey)
        XCTAssertEqual(JunoUserDefaults.hudPosition, .topCenter)
    }
}

// MARK: - JunoAppearancePreference (enum surface only; the setter and
// applyToSharedApplication touch NSApp and are deliberately not exercised).

final class AppearancePreferenceTests: JunoDefaultsRestoringTestCase {
    func testEnumCasesIdsAndTitles() {
        XCTAssertEqual(JunoAppearancePreference.allCases, [.light, .dark, .system])
        XCTAssertEqual(JunoAppearancePreference.light.title, "Light")
        XCTAssertEqual(JunoAppearancePreference.dark.title, "Dark")
        XCTAssertEqual(JunoAppearancePreference.system.title, "Match System")
        for preference in JunoAppearancePreference.allCases {
            XCTAssertEqual(preference.id, preference.rawValue)
        }
    }

    func testGetterDefaultsToLightWhenUnset() {
        XCTAssertEqual(JunoUserDefaults.appearancePreference, .light)
    }

    func testGetterReadsStoredRawValue() {
        // Write the raw value directly: the property setter calls
        // applyToSharedApplication() which touches NSApp.
        UserDefaults.standard.set("dark", forKey: JunoUserDefaults.appearancePreferenceKey)
        XCTAssertEqual(JunoUserDefaults.appearancePreference, .dark)
        UserDefaults.standard.set("system", forKey: JunoUserDefaults.appearancePreferenceKey)
        XCTAssertEqual(JunoUserDefaults.appearancePreference, .system)
    }

    func testGetterFallsBackToLightOnUnknownRawValue() {
        UserDefaults.standard.set("solarized", forKey: JunoUserDefaults.appearancePreferenceKey)
        XCTAssertEqual(JunoUserDefaults.appearancePreference, .light)
    }
}

// MARK: - Boolean toggle defaults

final class BooleanToggleDefaultTests: JunoDefaultsRestoringTestCase {
    func testDefaultsWhenUnset() {
        // ON by default
        XCTAssertTrue(JunoUserDefaults.hudDelightAnimationsEnabled)
        XCTAssertTrue(JunoUserDefaults.hudDelightSoundEnabled)
        XCTAssertTrue(JunoUserDefaults.hudShowDoneRowEnabled)
        XCTAssertTrue(JunoUserDefaults.hudOpenSoundEnabled)
        XCTAssertTrue(JunoUserDefaults.showInDock)
        XCTAssertTrue(JunoUserDefaults.actionsNotesSignatureEnabled)
        // OFF by default
        XCTAssertFalse(JunoUserDefaults.micVoiceProcessingEnabled)
        XCTAssertFalse(JunoUserDefaults.liveAdjudicationEnabled)
        XCTAssertFalse(JunoUserDefaults.actionsEnabled)
        XCTAssertFalse(JunoUserDefaults.developerModeEnabled)
        XCTAssertFalse(JunoUserDefaults.saveLogsToFileEnabled)
        XCTAssertFalse(JunoUserDefaults.onboardingCompleted)
        XCTAssertFalse(JunoUserDefaults.onboardingBrandDelightShown)
        XCTAssertFalse(JunoUserDefaults.actionsOnboardingDecisionMade)
        XCTAssertFalse(JunoUserDefaults.actionsNudgeShown)
    }

    func testOnByDefaultTogglesCanBeTurnedOff() {
        JunoUserDefaults.hudDelightAnimationsEnabled = false
        XCTAssertFalse(JunoUserDefaults.hudDelightAnimationsEnabled)
        JunoUserDefaults.hudDelightSoundEnabled = false
        XCTAssertFalse(JunoUserDefaults.hudDelightSoundEnabled)
        JunoUserDefaults.hudShowDoneRowEnabled = false
        XCTAssertFalse(JunoUserDefaults.hudShowDoneRowEnabled)
        JunoUserDefaults.showInDock = false
        XCTAssertFalse(JunoUserDefaults.showInDock)
        JunoUserDefaults.actionsNotesSignatureEnabled = false
        XCTAssertFalse(JunoUserDefaults.actionsNotesSignatureEnabled)
    }

    func testOffByDefaultTogglesCanBeTurnedOn() {
        JunoUserDefaults.micVoiceProcessingEnabled = true
        XCTAssertTrue(JunoUserDefaults.micVoiceProcessingEnabled)
        JunoUserDefaults.liveAdjudicationEnabled = true
        XCTAssertTrue(JunoUserDefaults.liveAdjudicationEnabled)
        JunoUserDefaults.actionsEnabled = true
        XCTAssertTrue(JunoUserDefaults.actionsEnabled)
        JunoUserDefaults.developerModeEnabled = true
        XCTAssertTrue(JunoUserDefaults.developerModeEnabled)
    }

    func testHudLiveTranscriptionsDefaultsToOff() {
        // Off when unset regardless of hardware eligibility.
        XCTAssertFalse(JunoUserDefaults.hudLiveTranscriptionsEnabled)
    }

    func testHudLiveTranscriptionsSetterIsGatedByEligibility() {
        // Machine-independent assertion: enabling only sticks on
        // preview-eligible hardware; on ineligible Macs it is forced off.
        JunoUserDefaults.hudLiveTranscriptionsEnabled = true
        XCTAssertEqual(
            JunoUserDefaults.hudLiveTranscriptionsEnabled,
            JunoPreviewEligibility.current.isEligible
        )
        JunoUserDefaults.hudLiveTranscriptionsEnabled = false
        XCTAssertFalse(JunoUserDefaults.hudLiveTranscriptionsEnabled)
    }
}

// MARK: - Onboarding completion + requirements version

final class OnboardingStateTests: JunoDefaultsRestoringTestCase {
    func testCompletingOnboardingStampsRequirementsVersion() {
        JunoUserDefaults.onboardingCompleted = true
        XCTAssertTrue(JunoUserDefaults.onboardingCompleted)
        XCTAssertEqual(
            JunoUserDefaults.onboardingRequirementsVersion,
            JunoUserDefaults.currentOnboardingRequirementsVersion
        )
    }

    func testUncompletingOnboardingDoesNotClearVersionStamp() {
        JunoUserDefaults.onboardingCompleted = true
        JunoUserDefaults.onboardingCompleted = false
        XCTAssertFalse(JunoUserDefaults.onboardingCompleted)
        XCTAssertEqual(
            JunoUserDefaults.onboardingRequirementsVersion,
            JunoUserDefaults.currentOnboardingRequirementsVersion
        )
    }

    func testResetOnboardingForRetestClearsState() {
        JunoUserDefaults.onboardingCompleted = true
        JunoUserDefaults.onboardingBrandDelightShown = true
        JunoUserDefaults.actionsOnboardingDecisionMade = true
        JunoUserDefaults.actionsNudgeShown = true
        JunoUserDefaults.incrementDictationCompletedCount()

        JunoUserDefaults.resetOnboardingForRetest()

        XCTAssertFalse(JunoUserDefaults.onboardingCompleted)
        XCTAssertEqual(JunoUserDefaults.onboardingRequirementsVersion, 0)
        XCTAssertFalse(JunoUserDefaults.onboardingBrandDelightShown)
        XCTAssertFalse(JunoUserDefaults.actionsOnboardingDecisionMade)
        XCTAssertFalse(JunoUserDefaults.actionsNudgeShown)
        XCTAssertEqual(JunoUserDefaults.dictationCompletedCount, 0)
    }
}

// MARK: - Dictation counter

final class DictationCounterTests: JunoDefaultsRestoringTestCase {
    func testCounterDefaultsToZero() {
        XCTAssertEqual(JunoUserDefaults.dictationCompletedCount, 0)
    }

    func testIncrementReturnsAndPersistsNewValue() {
        XCTAssertEqual(JunoUserDefaults.incrementDictationCompletedCount(), 1)
        XCTAssertEqual(JunoUserDefaults.incrementDictationCompletedCount(), 2)
        XCTAssertEqual(JunoUserDefaults.incrementDictationCompletedCount(), 3)
        XCTAssertEqual(JunoUserDefaults.dictationCompletedCount, 3)
    }

    func testIncrementFromExistingValue() {
        JunoUserDefaults.dictationCompletedCount = 41
        XCTAssertEqual(JunoUserDefaults.incrementDictationCompletedCount(), 42)
    }
}

// MARK: - Whisper preview defaults migration

final class WhisperPreviewMigrationTests: JunoDefaultsRestoringTestCase {
    func testFirstRunForcesLiveAdjudicationOffAndMarksMigrated() {
        UserDefaults.standard.set(true, forKey: JunoUserDefaults.liveAdjudicationEnabledKey)
        JunoUserDefaults.migrateWhisperPreviewDefaults()
        XCTAssertFalse(JunoUserDefaults.liveAdjudicationEnabled)
        XCTAssertTrue(UserDefaults.standard.bool(forKey: JunoUserDefaults.whisperPreviewDefaultsMigratedKey))
    }

    func testMigrationIsIdempotentAndRespectsLaterUserChoice() {
        JunoUserDefaults.migrateWhisperPreviewDefaults()
        // User opts back in after migration…
        JunoUserDefaults.liveAdjudicationEnabled = true
        // …and a relaunch's migration call must not clobber that choice.
        JunoUserDefaults.migrateWhisperPreviewDefaults()
        XCTAssertTrue(JunoUserDefaults.liveAdjudicationEnabled)
    }
}

// MARK: - Actions nudge dismissal cooldowns

final class ActionsNudgeDismissalTests: JunoDefaultsRestoringTestCase {
    func testNoDismissalRecordedByDefault() {
        for kind in JunoActionKind.allCases {
            XCTAssertNil(JunoUserDefaults.actionsNudgeDismissedAt(for: kind))
            XCTAssertFalse(JunoUserDefaults.dismissedRecently(for: kind))
        }
    }

    func testMarkDismissedStartsSevenDayCooldown() {
        JunoUserDefaults.markActionsNudgeDismissed(for: .note)
        XCTAssertNotNil(JunoUserDefaults.actionsNudgeDismissedAt(for: .note))
        XCTAssertTrue(JunoUserDefaults.dismissedRecently(for: .note))
        // Other kinds are independent.
        XCTAssertFalse(JunoUserDefaults.dismissedRecently(for: .reminder))
    }

    func testEightDayOldDismissalIsNotRecent() {
        let eightDaysAgo = Date().timeIntervalSince1970 - 8 * 24 * 3600
        UserDefaults.standard.set(eightDaysAgo, forKey: "JunoActionsNudgeDismissedAt_\(JunoActionKind.alarm.rawValue)")
        XCTAssertFalse(JunoUserDefaults.dismissedRecently(for: .alarm))
    }

    func testClearAllActionsNudgeDismissals() {
        for kind in JunoActionKind.allCases {
            JunoUserDefaults.markActionsNudgeDismissed(for: kind)
        }
        JunoUserDefaults.clearAllActionsNudgeDismissals()
        for kind in JunoActionKind.allCases {
            XCTAssertNil(JunoUserDefaults.actionsNudgeDismissedAt(for: kind))
        }
    }
}
