import Foundation

/// Pure decisions for interpreting post-paste Accessibility evidence.
///
/// Posting Cmd+V and proving that a custom editor accepted it are separate
/// capabilities. Electron surfaces can advertise a familiar text role while
/// returning an empty, stale, or placeholder AXValue. Treat those surfaces as
/// unverifiable so weak Accessibility evidence cannot become a false failure.
enum JunoPasteVerificationPolicy {
    private static let reliableRoles: Set<String> = [
        "AXTextField",
        "AXTextArea",
        "AXComboBox",
        "AXSearchField",
    ]

    private static let unreliableBundleIds: Set<String> = [
        "com.terminalx",
    ]

    static func isReadbackReliable(
        bundleId: String?,
        role: String,
        hasStringValue: Bool
    ) -> Bool {
        guard hasStringValue, reliableRoles.contains(role) else { return false }
        let normalizedBundleId = bundleId?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? ""
        return !unreliableBundleIds.contains(normalizedBundleId)
    }

    /// A correction monitor starts only after the delivery path has accepted
    /// the paste. Its snapshot is useful for learning later edits, but it is
    /// not stronger evidence than the delivery result and must not reopen the
    /// copy-ready HUD.
    static func shouldOfferCopyFallback(
        pasteWasAccepted: Bool,
        postPasteSnapshot: String
    ) -> Bool {
        guard !pasteWasAccepted else { return false }
        return postPasteSnapshot.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}
