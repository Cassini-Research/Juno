import AppKit
import SwiftUI

// MARK: - Privacy page
//
// The page makes one argument and then proves it.
//
//   1. Header + Journey Card establish the claim: voice → text, contained
//      inside "Your Mac." Nothing leaves.
//   2. "On this Mac, right now." is the receipt — live counts of what
//      Juno actually keeps, with per-category clear controls.
//   3. A quiet footer carries the policy link and version.
//
// Anything pricing-related does not belong here.

struct JunoPrivacyView: View {
    @StateObject private var retention = JunoRetentionSettingsModel()
    @State private var pendingClearHistory: Bool = false
    @State private var pendingDeleteRecordings: Bool = false
    @State private var policyButtonHover: Bool = false
    @Environment(\.colorScheme) private var scheme

    private let policyURL = URL(string: "https://usejuno.co/privacy")!

    var body: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: JunoUI.Spacing.l) {
                JunoPrivacyHeader()

                JunoPrivacyJourneyCard()

                JunoPrivacyOnThisMacPanel(
                    retention: retention,
                    onClearHistory: { pendingClearHistory = true },
                    onDeleteRecordings: { pendingDeleteRecordings = true }
                )

                JunoPrivacyFooter(
                    isHover: $policyButtonHover,
                    onTap: { NSWorkspace.shared.open(policyURL) }
                )
            }
            .padding(.horizontal, JunoTheme.PageInsets.detail)
            .padding(.top, JunoUI.Spacing.xl)
            .padding(.bottom, JunoUI.Spacing.xl)
            .frame(maxWidth: 720, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .top)
        .confirmationDialog(
            "Clear all dictation history?",
            isPresented: $pendingClearHistory,
            titleVisibility: .visible
        ) {
            Button("Clear History", role: .destructive) { retention.clearHistory() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This removes saved history entries from this Mac. It cannot be undone.")
        }
        .confirmationDialog(
            "Delete all retained recordings?",
            isPresented: $pendingDeleteRecordings,
            titleVisibility: .visible
        ) {
            Button("Delete Recordings", role: .destructive) { retention.pruneAllAudio() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This removes retained audio recordings used for replay and troubleshooting. It cannot be undone.")
        }
        .onAppear { retention.refresh() }
    }
}

// MARK: - Header

private struct JunoPrivacyHeader: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: JunoUI.Spacing.m) {
            HStack(spacing: JunoUI.Spacing.s) {
                Circle()
                    .fill(JunoUI.Calm.meadow)
                    .frame(width: 6, height: 6)
                    .overlay(
                        Circle()
                            .stroke(JunoUI.Calm.meadow.opacity(0.18), lineWidth: 4)
                    )
                Text("Privacy")
                    .junoType(.bodyEmphasis)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }

            HStack(alignment: .firstTextBaseline, spacing: 0) {
                Text("Everything stays on ")
                    .junoType(.display)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("your Mac.")
                    .junoType(.display)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(
                        RoundedRectangle(cornerRadius: 5, style: .continuous)
                            .fill(JunoUI.Calm.highlight.opacity(0.28))
                    )
            }
            .fixedSize(horizontal: false, vertical: true)

            Text("No cloud, no account, no tracking. Juno works the moment you open it.")
                .junoType(.body)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .frame(maxWidth: 440, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Journey card (the "Your Mac" boundary)

private struct JunoPrivacyJourneyCard: View {
    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var introActive: Bool = false
    @State private var settleWork: DispatchWorkItem?

    var body: some View {
        // The boundary tag sits OUTSIDE the clipShape so the offset(y: -10)
        // that lifts the capsule above the card edge isn't clipped away.
        VStack(alignment: .leading, spacing: 0) {
            cardBody
                .padding(.horizontal, JunoUI.Spacing.l)
                .padding(.top, JunoUI.Spacing.l + 4)
                .padding(.bottom, JunoUI.Spacing.m)
                .background(cardSurface)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(JunoTheme.border(scheme), lineWidth: 0.6)
                )
                .overlay(boundaryTag, alignment: .topLeading)
        }
        .onAppear {
            settleWork?.cancel()
            if reduceMotion {
                introActive = false
                return
            }
            introActive = true
            let work = DispatchWorkItem { introActive = false }
            settleWork = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.5, execute: work)
        }
        .onDisappear {
            settleWork?.cancel()
            settleWork = nil
            introActive = false
        }
    }

    private var cardSurface: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme))
            LinearGradient(
                colors: [
                    JunoUI.Calm.highlight.opacity(scheme == .dark ? 0.10 : 0.08),
                    Color.clear
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .blendMode(.plusLighter)
            .opacity(scheme == .dark ? 0.5 : 1)
        }
    }

    private var boundaryTag: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(JunoUI.Calm.meadow)
                .frame(width: 5, height: 5)
            Text("YOUR MAC")
                .junoType(.eyebrow)
                .foregroundStyle(JunoTheme.tertiaryText(scheme))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(
            Capsule().fill(JunoTheme.stageBackground(scheme))
        )
        .overlay(
            Capsule().strokeBorder(JunoTheme.border(scheme), lineWidth: 0.5)
        )
        .padding(.leading, JunoUI.Spacing.l)
        .offset(y: -10)
    }

    private var cardBody: some View {
        VStack(alignment: .leading, spacing: JunoUI.Spacing.m) {
            HStack(alignment: .center, spacing: JunoUI.Spacing.l) {
                PrivacyWaveBars(active: introActive)
                    .frame(width: 56, height: 32)

                PrivacyArrow()
                    .frame(width: 30, height: 8)
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))

                HStack(alignment: .firstTextBaseline, spacing: 0) {
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(JunoUI.Calm.highlight)
                        .frame(width: 2, height: 18)
                        .padding(.trailing, 10)
                    Text("Send the proposal to Sarah by Friday.")
                        .junoType(.bodyEmphasis)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                    PrivacyCursor(active: introActive)
                        .frame(width: 2, height: 14)
                        .foregroundStyle(JunoUI.Calm.highlight)
                        .padding(.leading, 2)
                }

                Spacer(minLength: 0)
            }

            JunoHairlineRule(.faint)

            (
                Text("Your voice becomes text right here. ")
                    .foregroundColor(JunoTheme.secondaryText(scheme))
                + Text("Nothing is uploaded.")
                    .foregroundColor(JunoTheme.primaryText(scheme))
            )
            .junoType(.caption)
            .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Wave + arrow + cursor primitives

private struct PrivacyWaveBars: View {
    var active: Bool
    @Environment(\.colorScheme) private var scheme

    private let heights: [CGFloat] = [7, 14, 24, 28, 18, 11, 6]
    private var highlighted: Set<Int> { [2, 3, 4] }
    private let baseAnimationStep: Double = 0.08

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0/30.0, paused: !active)) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            HStack(alignment: .center, spacing: 3) {
                ForEach(0..<heights.count, id: \.self) { i in
                    let phase = t * 2.2 + Double(i) * baseAnimationStep * 4
                    let scale = active ? (0.45 + 0.55 * (0.5 + 0.5 * sin(phase))) : 0.7
                    RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                        .fill(highlighted.contains(i) ? JunoUI.Calm.highlight : JunoTheme.primaryText(scheme))
                        .frame(width: 3, height: heights[i])
                        .scaleEffect(x: 1, y: max(0.25, CGFloat(scale)), anchor: .center)
                }
            }
        }
    }
}

private struct PrivacyArrow: View {
    var body: some View {
        GeometryReader { proxy in
            let h = proxy.size.height
            let w = proxy.size.width
            Path { path in
                path.move(to: CGPoint(x: 0, y: h / 2))
                path.addLine(to: CGPoint(x: w - 6, y: h / 2))
            }
            .stroke(Color.primary.opacity(0.5), style: StrokeStyle(lineWidth: 1, lineCap: .round))

            Path { path in
                path.move(to: CGPoint(x: w - 6, y: 0))
                path.addLine(to: CGPoint(x: w, y: h / 2))
                path.addLine(to: CGPoint(x: w - 6, y: h))
            }
            .fill(Color.primary.opacity(0.7))
        }
        .opacity(0.7)
    }
}

private struct PrivacyCursor: View {
    var active: Bool

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { context in
            let on = active
                ? (Int(context.date.timeIntervalSinceReferenceDate * 2) % 2 == 0)
                : true
            RoundedRectangle(cornerRadius: 1)
                .fill(Color.accentColor)
                .opacity(on ? 1.0 : 0.0)
        }
    }
}

// MARK: - On this Mac, right now
//
// One panel with two live rows. Each row turns the marketing claim into a
// receipt: what is actually saved on the user's Mac, and a button to remove
// just that category. The same `JunoRetentionSettingsModel` that powers the
// Settings → Storage row backs both rows so the numbers always match.

private struct JunoPrivacyOnThisMacPanel: View {
    @ObservedObject var retention: JunoRetentionSettingsModel
    let onClearHistory: () -> Void
    let onDeleteRecordings: () -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: JunoUI.Spacing.m) {
            VStack(alignment: .leading, spacing: 4) {
                Text("On this Mac, right now.")
                    .junoType(.subtitle)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(subtitleText)
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }

            VStack(spacing: 0) {
                row(
                    label: "Dictation history",
                    detail: historyDetail,
                    actionTitle: retention.isBusy ? "Clearing…" : "Clear",
                    actionEnabled: hasHistory && !retention.isBusy && retention.brokerReachable,
                    action: onClearHistory
                )
                JunoHairlineRule(.faint)
                row(
                    label: "Audio recordings",
                    detail: recordingsDetail,
                    actionTitle: retention.isBusy ? "Deleting…" : "Delete",
                    actionEnabled: hasRecordings && !retention.isBusy && retention.brokerReachable,
                    action: onDeleteRecordings
                )
            }
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(JunoTheme.elevatedCard(scheme))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(JunoTheme.border(scheme), lineWidth: 0.5)
            )

            if retention.brokerReachable, let totalMB = totalMegabytesOnDisk, totalMB > 0 {
                Text("Together using \(formatMegabytes(totalMB)) on this Mac.")
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
            }

            if let err = retention.inlineError {
                Text(err)
                    .junoType(.caption)
                    .foregroundStyle(JunoDesignTokens.danger)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var subtitleText: String {
        if !retention.brokerReachable {
            return "Juno is starting up. Counts will appear in a moment."
        }
        return "These are the only things Juno keeps. Clear them any time."
    }

    private var historyDetail: String {
        guard retention.brokerReachable else { return "—" }
        let entries = entryCount(forKey: "history_entries")
        if entries == 0 { return "Nothing saved yet." }
        return entries == 1 ? "1 entry" : "\(entries) entries"
    }

    private var recordingsDetail: String {
        guard retention.brokerReachable else { return "—" }
        let files = entryCount(forKey: "audio_files")
        if files == 0 { return "No recordings kept." }
        return files == 1 ? "1 recording" : "\(files) recordings"
    }

    private var hasHistory: Bool { entryCount(forKey: "history_entries") > 0 }
    private var hasRecordings: Bool { entryCount(forKey: "audio_files") > 0 }

    /// Pulls a count from the model's `storageSummaryLine`. The model keeps
    /// `storageStats` private, so we re-parse the line it formats from the
    /// raw broker payload.
    private func entryCount(forKey key: String) -> Int {
        guard let summary = retention.storageSummaryLine else { return 0 }
        switch key {
        case "history_entries":
            if let range = summary.range(of: #"(\d+)\s+history\s+entr"#, options: .regularExpression) {
                let token = summary[range].split(separator: " ").first.map(String.init) ?? ""
                return Int(token) ?? 0
            }
        case "audio_files":
            if let range = summary.range(of: #"(\d+)\s+recording"#, options: .regularExpression) {
                let token = summary[range].split(separator: " ").first.map(String.init) ?? ""
                return Int(token) ?? 0
            }
        default:
            return 0
        }
        return 0
    }

    /// Parses the aggregate "Using X.X MB" total out of the model's
    /// `storageSummaryLine`. The model keeps `storageStats` private, so we
    /// re-parse the formatted line it already emits. Returns `nil` if the
    /// summary is missing or the total can't be parsed.
    private var totalMegabytesOnDisk: Double? {
        guard let summary = retention.storageSummaryLine else { return nil }
        guard let range = summary.range(of: #"([0-9]+(?:\.[0-9]+)?)\s*MB"#, options: .regularExpression) else {
            return nil
        }
        let token = summary[range]
            .replacingOccurrences(of: "MB", with: "")
            .trimmingCharacters(in: .whitespaces)
        return Double(token)
    }

    private func formatMegabytes(_ mb: Double) -> String {
        String(format: "%.1f MB", mb)
    }

    @ViewBuilder
    private func row(
        label: String,
        detail: String,
        actionTitle: String,
        actionEnabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        HStack(alignment: .center, spacing: JunoUI.Spacing.m) {
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .junoType(.bodyEmphasis)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(detail)
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: JunoUI.Spacing.m)
            Button(actionTitle, action: action)
                .buttonStyle(JunoDestructiveSmallButtonStyle(scheme: scheme, isEnabled: actionEnabled))
                .disabled(!actionEnabled)
                .focusEffectDisabled()
        }
        .padding(.horizontal, JunoUI.Spacing.l)
        .padding(.vertical, JunoUI.Spacing.m)
    }
}

// MARK: - Footer

private struct JunoPrivacyFooter: View {
    @Binding var isHover: Bool
    let onTap: () -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(spacing: JunoUI.Spacing.s) {
            Text("Juno \(JunoProductIdentity.versionSummary)")
                .junoType(.caption)
                .foregroundStyle(JunoTheme.tertiaryText(scheme))

            Text("·")
                .junoType(.caption)
                .foregroundStyle(JunoTheme.tertiaryText(scheme))

            Button(action: onTap) {
                Text("Read the full privacy policy")
                    .junoType(.caption)
                    .foregroundStyle(isHover ? JunoTheme.primaryText(scheme) : JunoTheme.secondaryText(scheme))
                    .overlay(
                        Rectangle()
                            .fill(isHover ? JunoTheme.primaryText(scheme) : JunoTheme.border(scheme))
                            .frame(height: 1)
                            .offset(y: 2),
                        alignment: .bottom
                    )
            }
            .buttonStyle(.plain)
            .onHover { hovering in
                withAnimation(JunoUI.Motion.dim) { isHover = hovering }
            }

            Spacer(minLength: 0)
        }
        .padding(.top, JunoUI.Spacing.s)
    }
}
