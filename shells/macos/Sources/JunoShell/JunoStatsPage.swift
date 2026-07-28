import AppKit
import SwiftUI

struct JunoStatsPage: View {
    @StateObject private var stats = JunoStatsModel()
    @State private var range: JunoStatsRange = .sevenDays
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 18) {
                header
                rangePicker
                content
            }
            .junoDetailPagePadding()
        }
        .onAppear { stats.refresh() }
        .onReceive(
            NotificationCenter.default.publisher(
                for: NSApplication.didBecomeActiveNotification
            )
        ) { _ in
            stats.refresh()
        }
    }

    private var header: some View {
        JunoPageHeader(
            eyebrow: "Stats",
            title: "Your Juno activity",
            subtitle: "See how your dictation adds up over time. Everything stays private on this Mac.",
            trailing: {
                Button {
                    stats.refresh()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                        .font(.system(size: 12, weight: .semibold))
                }
                .junoSecondaryActionButton()
                .disabled(stats.isLoading)
                .help("Refresh stats")
            }
        )
    }

    private var rangePicker: some View {
        Picker("Time range", selection: $range) {
            ForEach(JunoStatsRange.allCases) { option in
                Text(option.title).tag(option)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(maxWidth: 360)
        .accessibilityLabel("Time range")
    }

    @ViewBuilder
    private var content: some View {
        if let period = stats.period(for: range) {
            let snapshot = JunoStatsDisplaySnapshot(
                period: period,
                range: range,
                lifetimeWords: JunoLifetimeWords.totalCount(),
                lifetimeDictations: JunoUserDefaults.dictationCompletedCount
            )
            if snapshot.totalWords == 0 {
                emptyState
            } else {
                VStack(alignment: .leading, spacing: 18) {
                    metrics(snapshot)
                    JunoStatsTrendCard(period: period, range: range)
                    JunoStatsTopAppsCard(
                        apps: period.topApps,
                        totalWords: period.totalWords
                    )
                    dataNote
                }
            }
        } else if stats.isLoading {
            loadingState
        } else if stats.lastError != nil {
            unavailableState
        } else {
            loadingState
        }
    }

    private func metrics(_ snapshot: JunoStatsDisplaySnapshot) -> some View {
        LazyVGrid(
            columns: Array(repeating: GridItem(.flexible(), spacing: 10), count: 3),
            alignment: .leading,
            spacing: 10
        ) {
            JunoStatsMetricCard(
                label: "Words",
                value: snapshot.totalWords.formatted(),
                symbol: "text.word.spacing"
            )
            JunoStatsMetricCard(
                label: "Time saved",
                value: Self.durationLabel(seconds: snapshot.timeSavedSeconds),
                symbol: "clock.arrow.circlepath"
            )
            JunoStatsMetricCard(
                label: "Dictations",
                value: snapshot.dictations.formatted(),
                symbol: "waveform"
            )
        }
    }

    private var loadingState: some View {
        HStack(spacing: 10) {
            ProgressView().controlSize(.small)
            Text("Loading your activity…")
                .junoType(.body)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
        }
        .frame(maxWidth: .infinity, minHeight: 220, alignment: .center)
        .junoPageCard()
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 30, weight: .light))
                .foregroundStyle(JunoDesignTokens.accent)
            Text("No activity in this range")
                .junoType(.bodyEmphasis)
                .foregroundStyle(JunoTheme.primaryText(scheme))
            Text(
                "Once you dictate with Juno, your words and time saved will appear here."
            )
                .junoType(.caption)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, minHeight: 220, alignment: .center)
        .junoPageCard()
    }

    private var unavailableState: some View {
        VStack(spacing: 10) {
            Image(systemName: "exclamationmark.arrow.triangle.2.circlepath")
                .font(.system(size: 26, weight: .regular))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
            Text("Stats are temporarily unavailable")
                .junoType(.bodyEmphasis)
                .foregroundStyle(JunoTheme.primaryText(scheme))
            Button("Try again") { stats.refresh() }
                .junoSecondaryActionButton()
        }
        .frame(maxWidth: .infinity, minHeight: 220, alignment: .center)
        .junoPageCard()
    }

    private var dataNote: some View {
        Text(
            range == .allTime
                ? "Lifetime words and dictations stay with this Juno install. The trend and app ranking reflect history currently kept on this Mac."
                : "Calculated from dictation history currently kept on this Mac."
        )
        .font(.system(size: 10.5, design: .rounded))
        .foregroundStyle(JunoTheme.tertiaryText(scheme))
        .fixedSize(horizontal: false, vertical: true)
    }

    static func durationLabel(seconds: Int) -> String {
        let seconds = max(0, seconds)
        if seconds < 60 {
            return "\(seconds) sec"
        }
        let minutes = seconds / 60
        if minutes < 60 {
            return "\(minutes) min"
        }
        let hours = minutes / 60
        let remainder = minutes % 60
        return remainder == 0 ? "\(hours) hr" : "\(hours) hr \(remainder) min"
    }
}

private struct JunoStatsMetricCard: View {
    let label: String
    let value: String
    let symbol: String
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(label)
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                Spacer(minLength: 4)
                Image(systemName: symbol)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(JunoDesignTokens.accent)
            }
            Text(value)
                .font(.system(size: 23, weight: .semibold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, minHeight: 70, alignment: .leading)
        .junoPageCard()
    }
}

private struct JunoStatsTrendCard: View {
    let period: StatsPeriodResponse
    let range: JunoStatsRange
    @Environment(\.colorScheme) private var scheme
    @State private var hoveredIndex: Int?

    private var values: [Int] {
        period.wordsByBucket.isEmpty ? [0] : period.wordsByBucket
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Words over time")
                        .junoType(.bodyEmphasis)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text(chartCaption)
                        .junoType(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
                Spacer(minLength: 12)
                Text(hoveredSummary)
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }

            GeometryReader { proxy in
                let peak = max(values.max() ?? 0, 1)
                HStack(alignment: .bottom, spacing: range == .sevenDays ? 10 : 5) {
                    ForEach(values.indices, id: \.self) { index in
                        let height = max(
                            3,
                            CGFloat(values[index]) / CGFloat(peak) * proxy.size.height
                        )
                        RoundedRectangle(cornerRadius: 4, style: .continuous)
                            .fill(
                                hoveredIndex == index || index == values.indices.last
                                    ? JunoDesignTokens.accent
                                    : JunoUI.Calm.barRest(scheme: scheme)
                            )
                            .frame(maxWidth: .infinity)
                            .frame(height: height)
                            .contentShape(Rectangle())
                            .onHover { hovering in
                                hoveredIndex = hovering ? index : nil
                            }
                            .help("\(bucketLabel(index)) · \(values[index].formatted()) words")
                    }
                }
            }
            .frame(height: 120)

            HStack {
                Text(edgeDateLabel(period.bucketStartDates.first))
                Spacer(minLength: 8)
                Text(edgeDateLabel(period.bucketEndDates.last))
            }
            .font(.system(size: 10, weight: .medium, design: .rounded))
            .foregroundStyle(JunoTheme.tertiaryText(scheme))
        }
        .junoPageCard()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Words over time")
        .accessibilityValue("\(period.totalWords) words")
    }

    private var chartCaption: String {
        switch range {
        case .sevenDays: return "Daily"
        case .thirtyDays: return "Every two days"
        case .allTime: return "Across your saved history"
        }
    }

    private var hoveredSummary: String {
        guard let index = hoveredIndex, values.indices.contains(index) else {
            return "\(period.totalWords.formatted()) words"
        }
        return "\(bucketLabel(index)) · \(values[index].formatted())"
    }

    private func bucketLabel(_ index: Int) -> String {
        guard period.bucketStartDates.indices.contains(index) else { return "Activity" }
        let start = displayDate(period.bucketStartDates[index], format: "MMM d")
        guard period.bucketEndDates.indices.contains(index) else { return start }
        let end = displayDate(period.bucketEndDates[index], format: "MMM d")
        return start == end ? start : "\(start)–\(end)"
    }

    private func edgeDateLabel(_ raw: String?) -> String {
        guard let raw else { return "" }
        return displayDate(raw, format: "MMM d, yyyy")
    }

    private func displayDate(_ raw: String, format: String) -> String {
        let parser = DateFormatter()
        parser.calendar = Calendar(identifier: .gregorian)
        parser.locale = Locale(identifier: "en_US_POSIX")
        parser.dateFormat = "yyyy-MM-dd"
        guard let date = parser.date(from: raw) else { return raw }
        let output = DateFormatter()
        output.locale = Locale.current
        output.setLocalizedDateFormatFromTemplate(format)
        return output.string(from: date)
    }
}

private struct JunoStatsTopAppsCard: View {
    let apps: [StatsAppResponse]
    let totalWords: Int
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Top apps")
                .junoType(.bodyEmphasis)
                .foregroundStyle(JunoTheme.primaryText(scheme))

            if apps.isEmpty {
                Text("App activity will appear after Juno has saved a dictation with app context.")
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .frame(maxWidth: .infinity, minHeight: 54, alignment: .leading)
            } else {
                ForEach(Array(apps.enumerated()), id: \.element.id) { index, app in
                    if index > 0 {
                        JunoHairlineRule(.faint)
                    }
                    HStack(spacing: 10) {
                        JunoStatsAppIcon(appName: app.name)
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(app.name)
                                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                                    .foregroundStyle(JunoTheme.primaryText(scheme))
                                    .lineLimit(1)
                                Spacer(minLength: 8)
                                Text("\(app.words.formatted()) words")
                                    .font(.system(size: 11, weight: .medium, design: .rounded))
                                    .monospacedDigit()
                                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                            }
                            ProgressView(
                                value: Double(app.words),
                                total: Double(max(totalWords, 1))
                            )
                            .progressViewStyle(.linear)
                            .tint(JunoDesignTokens.accent)
                        }
                    }
                    .padding(.vertical, 3)
                }
            }
        }
        .junoPageCard()
    }
}

private struct JunoStatsAppIcon: View {
    let appName: String

    var body: some View {
        Image(nsImage: resolveIcon())
            .resizable()
            .scaledToFit()
            .frame(width: 28, height: 28)
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .accessibilityHidden(true)
    }

    private func resolveIcon() -> NSImage {
        let workspace = NSWorkspace.shared
        let trimmed = appName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return fallbackIcon() }
        if trimmed.contains("."),
           let url = workspace.urlForApplication(withBundleIdentifier: trimmed) {
            return workspace.icon(forFile: url.path)
        }
        for path in [
            "/Applications/\(trimmed).app",
            "/System/Applications/\(trimmed).app",
            "/System/Applications/Utilities/\(trimmed).app",
        ] {
            if FileManager.default.fileExists(atPath: path) {
                return workspace.icon(forFile: path)
            }
        }
        return fallbackIcon()
    }

    private func fallbackIcon() -> NSImage {
        NSImage(systemSymbolName: "app", accessibilityDescription: nil) ?? NSImage()
    }
}
