import AppKit
import Foundation

/// Light is the product default (first-run polish); users can switch to dark or follow macOS in Settings.
enum JunoAppearancePreference: String, CaseIterable, Identifiable {
    case light
    case dark
    case system

    var id: String { rawValue }

    var title: String {
        switch self {
        case .light: return "Light"
        case .dark: return "Dark"
        case .system: return "Match System"
        }
    }

    func applyToSharedApplication() {
        switch self {
        case .light:
            NSApp.appearance = NSAppearance(named: .aqua)
        case .dark:
            NSApp.appearance = NSAppearance(named: .darkAqua)
        case .system:
            NSApp.appearance = nil
        }
    }
}

/// Central Juno-specific `UserDefaults` keys. Legacy Juno → Juno suite migration lives in ``JunoLegacyDefaultsMigration``.
enum JunoUserDefaults {
    static let onboardingCompletedKey = "JunoOnboardingCompleted"
    static let onboardingRequirementsVersionKey = "JunoOnboardingRequirementsVersion"
    static let currentOnboardingRequirementsVersion = 1
    static let preferredDisplayNameKey = "JunoPreferredDisplayName"
    static let onboardingBrandDelightShownKey = "JunoShellOnboardingBrandDelightShown"
    static let hudDelightAnimationsEnabledKey = "JunoHUDDelightAnimationsEnabled"
    static let hudDelightSoundEnabledKey = "JunoHUDDelightSoundEnabled"
    static let pauseSensitivitySecondsKey = "JunoPauseSensitivitySeconds"
    static let hudPositionKey = "JunoHUDPosition"
    static let hudLiveTranscriptionsEnabledKey = "JunoHUDLiveTranscriptionsEnabled"
    static let liveAdjudicationEnabledKey = "JunoLiveAdjudicationEnabled"
    static let whisperPreviewDefaultsMigratedKey = "JunoWhisperPreviewDefaultsMigrated"
    static let showInDockKey = "JunoShowInDock"
    static let micVoiceProcessingEnabledKey = "JunoMicVoiceProcessingEnabled"
    static let hudShowDoneRowEnabledKey = "JunoHUDShowDoneRowEnabled"
    static let languageModeKey = "JunoLanguageMode"
    static let developerModeEnabledKey = "JunoDeveloperModeEnabled"
    static let saveLogsToFileEnabledKey = "JunoSaveLogsToFileEnabled"
    static let appearancePreferenceKey = "JunoAppearancePreference"
    /// Top-level enable for the Voice Actions feature (notes & reminders).
    /// Defaults to ``false`` — users opt in explicitly via Settings or by
    /// confirming the Home-page nudge. When off, action utterances paste
    /// as today and no reminders/notes are created.
    static let actionsEnabledKey = "JunoActionsEnabled"
    /// Append a "Captured with Juno · {time}" line to notes saved through
    /// the Actions feature. Defaults to ``true``; users can turn it off
    /// in Settings → Voice Actions. Notes-only — never on dictation
    /// pasted into other apps.
    static let actionsNotesSignatureEnabledKey = "JunoActionsNotesSignatureEnabled"
    /// Defaults to **light** when unset so new installs match the first-run / onboarding aesthetic.
    static var appearancePreference: JunoAppearancePreference {
        get {
            if UserDefaults.standard.object(forKey: appearancePreferenceKey) == nil {
                return .light
            }
            let raw = UserDefaults.standard.string(forKey: appearancePreferenceKey) ?? JunoAppearancePreference.light.rawValue
            return JunoAppearancePreference(rawValue: raw) ?? .light
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: appearancePreferenceKey)
            newValue.applyToSharedApplication()
        }
    }

    static var onboardingCompleted: Bool {
        get { UserDefaults.standard.bool(forKey: onboardingCompletedKey) }
        set {
            UserDefaults.standard.set(newValue, forKey: onboardingCompletedKey)
            if newValue {
                UserDefaults.standard.set(currentOnboardingRequirementsVersion, forKey: onboardingRequirementsVersionKey)
            }
        }
    }

    static var onboardingRequirementsVersion: Int {
        get { UserDefaults.standard.integer(forKey: onboardingRequirementsVersionKey) }
        set { UserDefaults.standard.set(newValue, forKey: onboardingRequirementsVersionKey) }
    }

    /// Trimming empty; persisted for home greetings.
    static var preferredDisplayName: String? {
        get {
            let s = UserDefaults.standard.string(forKey: preferredDisplayNameKey)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !s.isEmpty && Self.meaningfulNameCharacterCount(s) < 3 {
                return nil
            }
            return s.isEmpty ? nil : s
        }
        set {
            let t = newValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if t.isEmpty || Self.meaningfulNameCharacterCount(t) < 3 {
                UserDefaults.standard.removeObject(forKey: preferredDisplayNameKey)
            } else {
                UserDefaults.standard.set(t, forKey: preferredDisplayNameKey)
            }
        }
    }

    static func meaningfulNameCharacterCount(_ value: String) -> Int {
        value.unicodeScalars.reduce(0) { count, scalar in
            CharacterSet.alphanumerics.contains(scalar) ? count + 1 : count
        }
    }

    static var onboardingBrandDelightShown: Bool {
        get { UserDefaults.standard.bool(forKey: onboardingBrandDelightShownKey) }
        set {
            UserDefaults.standard.set(newValue, forKey: onboardingBrandDelightShownKey)
        }
    }

    /// Visual micro-delights in the HUD (paste/copy confirmations, subtle sweeps).
    /// Defaults to ON.
    static var hudDelightAnimationsEnabled: Bool {
        get {
            let ud = UserDefaults.standard
            if ud.object(forKey: hudDelightAnimationsEnabledKey) == nil { return true }
            return ud.bool(forKey: hudDelightAnimationsEnabledKey)
        }
        set {
            UserDefaults.standard.set(newValue, forKey: hudDelightAnimationsEnabledKey)
        }
    }

    /// Small tick sound for paste/copy confirmations in the HUD.
    /// Defaults to ON.
    static var hudDelightSoundEnabled: Bool {
        get {
            let ud = UserDefaults.standard
            if ud.object(forKey: hudDelightSoundEnabledKey) == nil { return true }
            return ud.bool(forKey: hudDelightSoundEnabledKey)
        }
        set {
            UserDefaults.standard.set(newValue, forKey: hudDelightSoundEnabledKey)
        }
    }

    /// Show the "+N words placed" confirmation row in the HUD after a paste.
    /// Defaults to ON.
    static var hudShowDoneRowEnabled: Bool {
        get {
            let ud = UserDefaults.standard
            if ud.object(forKey: hudShowDoneRowEnabledKey) == nil { return true }
            return ud.bool(forKey: hudShowDoneRowEnabledKey)
        }
        set { UserDefaults.standard.set(newValue, forKey: hudShowDoneRowEnabledKey) }
    }

    /// Pause time (seconds) before a mid-dictation partial paste triggers.
    /// Defaults to 1.4s, clamped to 0.8–3.0s.
    static var pauseSensitivitySeconds: Double {
        get {
            let ud = UserDefaults.standard
            if ud.object(forKey: pauseSensitivitySecondsKey) == nil { return 1.4 }
            let v = ud.double(forKey: pauseSensitivitySecondsKey)
            return min(3.0, max(0.8, v))
        }
        set {
            UserDefaults.standard.set(min(3.0, max(0.8, newValue)), forKey: pauseSensitivitySecondsKey)
        }
    }

    enum HUDPosition: String, CaseIterable, Identifiable {
        case topCenter = "top_center"
        case bottomCenter = "bottom_center"
        var id: String { rawValue }
        var title: String {
            switch self {
            case .topCenter: return "Top center"
            case .bottomCenter: return "Bottom center"
            }
        }
    }

    /// Default is **top center**. Legacy `cursor_follow` raw values from earlier
    /// builds are migrated to `topCenter` on first read so retired option doesn't
    /// strand existing users.
    static var hudPosition: HUDPosition {
        get {
            let raw = UserDefaults.standard.string(forKey: hudPositionKey)
            if raw == "cursor_follow" {
                return .topCenter
            }
            return HUDPosition(rawValue: raw ?? "") ?? .topCenter
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: hudPositionKey)
        }
    }

    /// Live transcriptions in the HUD. Defaults to OFF. When ON the floating
    /// HUD shows partial transcript text as you speak. When OFF, Juno still
    /// shows the listening/refining HUD but sends no preview audio chunks.
    static var hudLiveTranscriptionsEnabled: Bool {
        get {
            let ud = UserDefaults.standard
            guard JunoPreviewEligibility.current.isEligible else { return false }
            if ud.object(forKey: hudLiveTranscriptionsEnabledKey) == nil { return false }
            return ud.bool(forKey: hudLiveTranscriptionsEnabledKey)
        }
        set {
            let allowed = !newValue || JunoPreviewEligibility.current.isEligible
            UserDefaults.standard.set(allowed ? newValue : false, forKey: hudLiveTranscriptionsEnabledKey)
        }
    }

    /// Run model adjudication on in-speech snapshots while the user is still
    /// dictating. The Whisper-driven HUD should move forward append-only by
    /// default; mid-speech rewrites are opt-in because they can make visible
    /// words jump before final delivery settles the transcript.
    static var liveAdjudicationEnabled: Bool {
        get {
            let ud = UserDefaults.standard
            if ud.object(forKey: liveAdjudicationEnabledKey) == nil { return false }
            return ud.bool(forKey: liveAdjudicationEnabledKey)
        }
        set { UserDefaults.standard.set(newValue, forKey: liveAdjudicationEnabledKey) }
    }

    static func migrateWhisperPreviewDefaults() {
        let ud = UserDefaults.standard
        guard ud.object(forKey: whisperPreviewDefaultsMigratedKey) == nil else { return }
        ud.set(false, forKey: liveAdjudicationEnabledKey)
        ud.set(true, forKey: whisperPreviewDefaultsMigratedKey)
    }

    static let hudOpenSoundEnabledKey = "JunoHUDOpenSoundEnabled"

    /// System "Tink" plays the moment the dictation HUD appears.
    /// Defaults to ON.
    static var hudOpenSoundEnabled: Bool {
        get {
            let ud = UserDefaults.standard
            if ud.object(forKey: hudOpenSoundEnabledKey) == nil { return true }
            return ud.bool(forKey: hudOpenSoundEnabledKey)
        }
        set { UserDefaults.standard.set(newValue, forKey: hudOpenSoundEnabledKey) }
    }

    /// Menu-bar-first. Defaults to showing a Dock icon (regular macOS app).
    static var showInDock: Bool {
        get {
            let ud = UserDefaults.standard
            if ud.object(forKey: showInDockKey) == nil { return true }
            return ud.bool(forKey: showInDockKey)
        }
        set {
            UserDefaults.standard.set(newValue, forKey: showInDockKey)
        }
    }

    /// Toggle AVAudioEngine's voice processing (noise suppression / AGC).
    ///
    /// Defaults to OFF. Voice processing has been the proximate cause of two
    /// independent fresh-install failures we've debugged:
    ///
    /// 1. ``setVoiceProcessingEnabled(true)`` produced all-zero PCM frames
    ///    on a Mac14,5 / 25.x build — captured WAVs had RMS=0, peak=0
    ///    across every sample. mlx_whisper hallucinated Portuguese loops.
    /// 2. On the example user's 2026-04-29 support bundle the captured audio wasn't
    ///    silent but was distorted enough that whisper hallucinated
    ///    ``"www. alberalalalal..."`` loops.
    ///
    /// Off-by-default lets users in noisy environments opt in, while
    /// users on Macs where the macOS voice-processing front-end is
    /// broken get working dictation out of the box. The Settings →
    /// Audio → "Mic processing" toggle still controls the value at
    /// runtime; this only changes what fresh installs (and existing
    /// installs that never touched the toggle) inherit.
    static var micVoiceProcessingEnabled: Bool {
        get {
            let ud = UserDefaults.standard
            if ud.object(forKey: micVoiceProcessingEnabledKey) == nil { return false }
            return ud.bool(forKey: micVoiceProcessingEnabledKey)
        }
        set {
            UserDefaults.standard.set(newValue, forKey: micVoiceProcessingEnabledKey)
        }
    }

    static var languageMode: String {
        get {
            let raw = UserDefaults.standard.string(forKey: languageModeKey) ?? "auto"
            let allowed: Set<String> = ["auto", "en", "pair:en,hi", "zh", "es", "keep_original"]
            return allowed.contains(raw) ? raw : "auto"
        }
        set {
            UserDefaults.standard.set(newValue, forKey: languageModeKey)
        }
    }

    /// Hides Advanced/Developer settings from the default Settings view. When ON,
    /// the "Developer settings" link appears at the bottom of Settings and exposes
    /// diagnostics + the support-bundle exporter. Defaults to OFF.
    static var developerModeEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: developerModeEnabledKey) }
        set { UserDefaults.standard.set(newValue, forKey: developerModeEnabledKey) }
    }

    /// When ON, redirects the Juno app's stderr (where NSLog writes) into
    /// ~/Library/Logs/Juno/juno-app.log so support bundles include shell-side
    /// log lines. Takes effect on next launch — the redirect is installed once
    /// during app init. Defaults to OFF.
    static var saveLogsToFileEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: saveLogsToFileEnabledKey) }
        set { UserDefaults.standard.set(newValue, forKey: saveLogsToFileEnabledKey) }
    }

    /// Voice Actions master toggle. Defaults to OFF — opt-in.
    static var actionsEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: actionsEnabledKey) }
        set { UserDefaults.standard.set(newValue, forKey: actionsEnabledKey) }
    }

    /// "Captured with Juno · {time}" signature on saved notes. Defaults to ON.
    static var actionsNotesSignatureEnabled: Bool {
        get {
            let ud = UserDefaults.standard
            if ud.object(forKey: actionsNotesSignatureEnabledKey) == nil { return true }
            return ud.bool(forKey: actionsNotesSignatureEnabledKey)
        }
        set { UserDefaults.standard.set(newValue, forKey: actionsNotesSignatureEnabledKey) }
    }

    /// Onboarding's Voice Actions step ("Juno can do things too") records
    /// whether the user tapped **Enable Voice Actions** (true) or **Maybe
    /// later** / completed onboarding before the step existed (false). The
    /// gentle post-onboarding nudge (see ``actionsNudgeShownKey``) only
    /// surfaces when this is ``false`` — i.e. the user deferred during
    /// onboarding and we want to remind them once after a few dictations.
    static let actionsOnboardingDecisionMadeKey = "JunoActionsOnboardingDecisionMade"
    static var actionsOnboardingDecisionMade: Bool {
        get { UserDefaults.standard.bool(forKey: actionsOnboardingDecisionMadeKey) }
        set { UserDefaults.standard.set(newValue, forKey: actionsOnboardingDecisionMadeKey) }
    }

    /// Cumulative count of successful dictations the user has completed
    /// (pasted text + non-empty transcript). Drives the one-time
    /// post-onboarding Voice Actions nudge — see
    /// ``actionsNudgeShownKey``. Persists for the lifetime of the install
    /// independent of ``JunoLifetimeWords`` because lifetime-words counts
    /// words and we want a count of completed *sessions*.
    static let dictationCompletedCountKey = "JunoDictationCompletedCount"
    static var dictationCompletedCount: Int {
        get { UserDefaults.standard.integer(forKey: dictationCompletedCountKey) }
        set { UserDefaults.standard.set(newValue, forKey: dictationCompletedCountKey) }
    }

    /// Increment the dictation counter. Returns the new value. Called from
    /// the Shell on every successful paste so the nudge can fire on dictation #3.
    @discardableResult
    static func incrementDictationCompletedCount() -> Int {
        let next = dictationCompletedCount + 1
        dictationCompletedCount = next
        return next
    }

    /// "Try saying 'Juno, take a note...'" nudge state — fires exactly
    /// once after the third successful dictation when the user deferred
    /// Voice Actions during onboarding. Set to true the moment the nudge
    /// appears (not when dismissed) so a flap-on-relaunch can't show it
    /// a second time.
    static let actionsNudgeShownKey = "JunoActionsNudgeShown"
    static var actionsNudgeShown: Bool {
        get { UserDefaults.standard.bool(forKey: actionsNudgeShownKey) }
        set { UserDefaults.standard.set(newValue, forKey: actionsNudgeShownKey) }
    }

    /// Clears onboarding completion and first-run delight flags so the welcome flow
    /// and permission cards behave like a fresh install. macOS TCC entries are
    /// unchanged — use `scripts/reset_juno_tcc.sh` or System Settings to revoke.
    static func resetOnboardingForRetest() {
        let ud = UserDefaults.standard
        ud.set(false, forKey: onboardingCompletedKey)
        ud.removeObject(forKey: onboardingRequirementsVersionKey)
        ud.removeObject(forKey: onboardingBrandDelightShownKey)
        ud.removeObject(forKey: actionsOnboardingDecisionMadeKey)
        ud.removeObject(forKey: actionsNudgeShownKey)
        ud.removeObject(forKey: dictationCompletedCountKey)
        ud.synchronize()
    }
}
