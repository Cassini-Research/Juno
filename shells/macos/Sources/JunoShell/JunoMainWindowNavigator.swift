import Combine
import Foundation

/// Shared sidebar selection so `JunoMainWindow.show(..., section:)` updates
/// the open window instead of only fronting a stale pane.
@MainActor
final class JunoMainWindowNavigator: ObservableObject {
    static let shared = JunoMainWindowNavigator()

    @Published var section: MainSidebar = .home

    /// When set, App rules (`SurfacePresetsView`) pre-selects this bundle on next appear.
    @Published var pendingPresetBundleId: String?

    /// `MemoryManagementView.Category` raw value, e.g. `vocab`.
    @Published var pendingMemoryCategoryRaw: String?

    /// Prefills the vocabulary term field when opening Snippets & Memory.
    @Published var pendingMemoryVocabPrefill: String?

    /// When set, History pre-selects this ``UtteranceHistoryEntry/utteranceId`` on next successful load.
    @Published var pendingHistoryUtteranceId: String?

    private init() {}

    /// Deep link from Home or History into App rules with an optional bundle id.
    func openAppRules(prefillBundleId: String?) {
        let trimmed = (prefillBundleId ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        pendingPresetBundleId = trimmed.isEmpty ? nil : trimmed
        section = .surfacePresets
    }

    /// Deep link into Snippets & Memory with optional tab and vocabulary prefill.
    func openDictionaryAndMemory(categoryRaw: String = "vocab", vocabPrefill: String? = nil) {
        pendingMemoryCategoryRaw = categoryRaw
        let t = (vocabPrefill ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        pendingMemoryVocabPrefill = t.isEmpty ? nil : t
        section = .personalization
    }

    /// Open History, optionally focusing a specific utterance row (e.g. from Home recent dictations).
    func openHistory(utteranceId: String?) {
        let t = (utteranceId ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        pendingHistoryUtteranceId = t.isEmpty ? nil : t
        section = .history
    }
}
