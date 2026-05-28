// JunoSettingsToastCenter.swift
// Audit fix C2 (RC-4 / S-S1): centralized broker-error surface for Settings.
//
// Background: per-setting `inlineError` strings exist and DO get set on
// broker failures, but they appear in small caption text near the failing
// control. When the broker is offline and a user toggles three settings in
// quick succession, only the most recently visible inline error is salient,
// and switching cards hides earlier ones. A single top-of-page toast that
// any setting model can post to is the simpler, more robust pattern.
//
// Usage from a settings @ObservableObject after a broker POST fails:
//
//     JunoBroker.postJSON(path: "...", payload: ...) { obj in
//         if (obj["ok"] as? Bool) == false {
//             let msg = (obj["error"] as? String) ?? "Could not save"
//             self.inlineError = msg                          // existing
//             JunoSettingsToastCenter.shared.report(msg)      // new
//         }
//     }

import Foundation
import SwiftUI

@MainActor
final class JunoSettingsToastCenter: ObservableObject {
    static let shared = JunoSettingsToastCenter()

    struct Entry: Identifiable, Equatable {
        let id = UUID()
        let message: String
        let severity: Severity
    }

    enum Severity {
        case error
        case info
    }

    @Published private(set) var current: Entry?

    private var dismissTask: Task<Void, Never>?

    private init() {}

    /// Post an error message to the top-of-Settings toast. Replaces any prior
    /// entry; auto-dismisses after `autoDismissAfter` seconds.
    func report(_ message: String, severity: Severity = .error, autoDismissAfter seconds: Double = 6) {
        guard !message.isEmpty else { return }
        let entry = Entry(message: message, severity: severity)
        current = entry
        dismissTask?.cancel()
        dismissTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self else { return }
                if self.current?.id == entry.id {
                    self.current = nil
                }
            }
        }
    }

    func dismiss() {
        dismissTask?.cancel()
        current = nil
    }
}

/// View placed at the top of the Settings page — renders the toast when set.
/// Uses the same design tokens as the rest of the Juno chrome.
struct JunoSettingsToastBanner: View {
    @ObservedObject private var center = JunoSettingsToastCenter.shared
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        if let entry = center.current {
            HStack(spacing: 10) {
                Image(systemName: entry.severity == .error
                      ? "exclamationmark.triangle.fill"
                      : "info.circle.fill")
                    .foregroundStyle(entry.severity == .error
                                     ? JunoDesignTokens.danger
                                     : JunoTheme.secondaryText(scheme))
                Text(entry.message)
                    .font(.system(.callout, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                Button {
                    center.dismiss()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Dismiss")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(entry.severity == .error
                          ? JunoDesignTokens.danger.opacity(0.10)
                          : JunoTheme.cardBackground(scheme).opacity(0.85))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(
                        entry.severity == .error
                            ? JunoDesignTokens.danger.opacity(0.35)
                            : JunoTheme.border(scheme).opacity(0.6),
                        lineWidth: 0.5
                    )
            )
            .transition(.opacity.combined(with: .move(edge: .top)))
            .animation(.easeOut(duration: 0.18), value: entry.id)
            .padding(.horizontal, 24)
            .padding(.bottom, 8)
        }
    }
}
