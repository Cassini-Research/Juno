import SwiftUI

struct JunoScreenContextHomeCard: View {
    @ObservedObject var stats: JunoStatsModel

    @ObservedObject private var windowNav = JunoMainWindowNavigator.shared
    @AppStorage(JunoUserDefaults.screenContextEnabledKey) private var enabled = false
    @Environment(\.colorScheme) private var scheme

    @State private var permissionGranted = JunoScreenContextAccess.permissionGranted
    @State private var nudgeDismissedRecently = JunoUserDefaults.screenContextNudgeDismissedRecently

    var body: some View {
        Group {
            if let state {
                cueWithDivider {
                    cueRow(for: state)
                }
                .onAppear {
                    permissionGranted = JunoScreenContextAccess.permissionGranted
                    nudgeDismissedRecently = JunoUserDefaults.screenContextNudgeDismissedRecently
                }
            }
        }
    }

    private enum CardState {
        case enable
        case finishSetup
    }

    private var state: CardState? {
        guard JunoUserDefaults.onboardingCompleted else { return nil }
        if enabled {
            return permissionGranted ? nil : .finishSetup
        }
        guard usageWordCount >= 300 else { return nil }
        guard !nudgeDismissedRecently else { return nil }
        return .enable
    }

    private var usageWordCount: Int {
        max(stats.wordsWeek ?? 0, stats.wordsToday ?? 0, JunoLifetimeWords.totalCount())
    }

    @ViewBuilder
    private func cueWithDivider<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack(spacing: 0) {
            content()
                .padding(.horizontal, JunoTheme.PageInsets.detail)
                .padding(.vertical, 10)
                .frame(minHeight: 56, alignment: .center)
            JunoHairlineRule(.faint)
        }
    }

    private func cueRow(for state: CardState) -> some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: "viewfinder")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(JunoTheme.tertiaryText(scheme))
                .frame(width: 18, height: 18)

            VStack(alignment: .leading, spacing: 2) {
                Text("VISIBLE TEXT")
                    .junoType(.eyebrow)
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
                Text(rowCopy(for: state))
                    .font(.system(size: 12.5, weight: .medium, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .lineLimit(2)
            }

            Spacer(minLength: 8)

            Button(ctaTitle(for: state)) {
                handleCTA(for: state)
            }
            .buttonStyle(JunoScreenContextCTAButtonStyle())
            .focusable(false)

            if state == .enable {
                Button {
                    JunoUserDefaults.markScreenContextNudgeDismissed()
                    nudgeDismissedRecently = true
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .semibold))
                }
                .buttonStyle(.plain)
                .focusable(false)
                .foregroundStyle(JunoTheme.tertiaryText(scheme))
                .frame(width: 18, height: 18)
                .help("Not now")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func rowCopy(for state: CardState) -> String {
        switch state {
        case .enable:
            return "Spell on-screen names and code terms right by reading your screen with on-device OCR. Nothing is stored or sent anywhere, and it only runs while you dictate."
        case .finishSetup:
            return "Turn on Juno in macOS Screen Recording so it can read on-screen terms locally while you dictate."
        }
    }

    private func ctaTitle(for state: CardState) -> String {
        switch state {
        case .enable: return "Open Settings"
        case .finishSetup: return "Open Settings"
        }
    }

    private func handleCTA(for state: CardState) {
        switch state {
        case .enable:
            enabled = true
            JunoUserDefaults.screenContextEnabled = true
            requestPermission()
        case .finishSetup:
            requestPermission()
        }
    }

    private func requestPermission() {
        JunoScreenContextAccess.requestFromExplicitUserAction { granted in
            permissionGranted = granted
        }
    }
}

private struct JunoScreenContextCTAButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 11, weight: .semibold, design: .rounded))
            .foregroundStyle(.white)
            .padding(.horizontal, 11)
            .padding(.vertical, 4)
            .background(
                Capsule(style: .continuous)
                    .fill(JunoDesignTokens.accent.opacity(configuration.isPressed ? 0.84 : 1))
            )
            .opacity(isEnabled ? 1 : 0.45)
    }
}
