// JunoHomeStatsGraph.swift
//
// One consolidated stats section for Home. Replaces the four-cell layout
// (words today / time saved / week / apps today) with a single rich graph:
//
//   anchor (TODAY · 82 words · "37s saved · most in Notes")
//     │
//     └── 7-day bar chart on the right (today is the navy emphasis bar)
//             │
//             └── foot strip: "Today across [icons] +N · 412 words this week"
//
// All copy lives in this file; no string keys. Real app icons resolved via
// NSWorkspace by app name (matches Recent Dictations rows).

import AppKit
import SwiftUI

struct JunoHomeStatsGraph: View {
    @ObservedObject var stats: JunoStatsModel
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .bottom, spacing: 24) {
                anchor
                    .frame(minWidth: 120, alignment: .leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                chart
                    .frame(maxWidth: .infinity)
            }
            footStrip
        }
        .padding(.horizontal, JunoTheme.PageInsets.detail)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Anchor (left column)

    private var anchor: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Today")
                .junoType(.eyebrow)
                .foregroundStyle(JunoTheme.tertiaryText(scheme))
            HStack(alignment: .lastTextBaseline, spacing: 6) {
                Text(stats.wordsToday.map(String.init) ?? "—")
                    .font(.system(size: 32, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("words")
                    .font(.system(size: 11.5, weight: .medium, design: .rounded))
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
            }
            anchorCaption
                .padding(.top, 4)
        }
    }

    @ViewBuilder
    private var anchorCaption: some View {
        let savedSec = stats.timeSavedSec ?? 0
        let savedMin = stats.timeSavedMin ?? 0
        let savedLabel: String? = {
            if savedSec > 0 { return savedSec >= 60 ? "\(savedMin)m saved" : "\(savedSec)s saved" }
            return nil
        }()
        let topApp = stats.topAppToday

        HStack(spacing: 4) {
            if let savedLabel {
                Text(savedLabel)
                    .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoUI.Calm.meadow)
            }
            if savedLabel != nil && topApp != nil {
                Text("·")
                    .font(.system(size: 10.5, design: .rounded))
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
            }
            if let topApp {
                Text("most in ")
                    .font(.system(size: 10.5, design: .rounded))
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
                +
                Text(topApp)
                    .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                    .foregroundColor(JunoTheme.secondaryText(scheme))
            } else if savedLabel == nil {
                Text("Start dictating to see your day take shape.")
                    .font(.system(size: 10.5, design: .rounded))
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
            }
        }
    }

    // MARK: - 7-day bar chart

    private var chart: some View {
        let values = stats.wordsByDay ?? [0, 0, 0, 0, 0, 0, 0]
        let labels = sevenDayLabels()
        let maxValue = max(values.max() ?? 0, 1) // avoid div-by-zero
        return HStack(alignment: .bottom, spacing: 6) {
            ForEach(0..<7, id: \.self) { i in
                DayColumn(
                    value: i < values.count ? values[i] : 0,
                    maxValue: maxValue,
                    label: i < labels.count ? labels[i] : "",
                    isToday: i == 6,
                    scheme: scheme
                )
            }
        }
        .frame(height: 96)
        .accessibilityElement()
        .accessibilityLabel("Words spoken, last 7 days")
    }

    /// `["Mon","Tue","Wed","Thu","Fri","Sat","Today"]` aligned so the 7th
    /// slot is always Today. Uses `Calendar.current.shortWeekdaySymbols`.
    private func sevenDayLabels() -> [String] {
        let cal = Calendar.current
        let today = Date()
        var out: [String] = []
        for offset in (0...6).reversed() { // 6, 5, ..., 0
            if let d = cal.date(byAdding: .day, value: -offset, to: today) {
                if offset == 0 {
                    out.append("Today")
                } else {
                    let weekday = cal.component(.weekday, from: d) // 1...7
                    let symbols = cal.shortWeekdaySymbols // [Sun, Mon, …]
                    out.append(symbols[(weekday - 1) % symbols.count])
                }
            } else {
                out.append("")
            }
        }
        return out
    }

    // MARK: - Foot strip

    private var footStrip: some View {
        let allApps = stats.appsTodayTop ?? []
        let visible = Array(allApps.prefix(3))
        let overflow = max(0, allApps.count - visible.count)
        let weekWords = stats.wordsWeek ?? 0

        // Always reserve a stable row height (≥ 18pt) so the foot strip
        // doesn't change size when there are no apps yet — the entire page
        // would otherwise jump on first dictation. Both content branches
        // render their text in the same visual rhythm.
        return HStack(spacing: 8) {
            if visible.isEmpty {
                Text("No apps yet today")
                    .font(.system(size: 10.5, design: .rounded))
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
            } else {
                Text("Today across")
                    .font(.system(size: 10.5, design: .rounded))
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
                HStack(spacing: 4) {
                    ForEach(visible, id: \.self) { name in
                        AppIconChip(appName: name)
                    }
                    if overflow > 0 {
                        Text("+\(overflow)")
                            .font(.system(size: 9.5, weight: .semibold, design: .rounded))
                            .monospacedDigit()
                            .padding(.horizontal, 6)
                            .frame(height: 16)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(JunoUI.hairline(.faint, scheme: scheme))
                            )
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .help(allApps.dropFirst(visible.count).joined(separator: ", "))
                    }
                }
            }
            // Push the week count to the trailing edge so it uses the
            // empty space on the right instead of crowding the app icons
            // on the left.
            Spacer(minLength: 12)
            if weekWords > 0 {
                (Text("\(weekWords) words")
                    .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                    .foregroundColor(JunoTheme.secondaryText(scheme))
                +
                Text(" this week")
                    .font(.system(size: 10.5, design: .rounded))
                    .foregroundColor(JunoTheme.tertiaryText(scheme)))
            } else {
                Text("Quiet week so far")
                    .font(.system(size: 10.5, design: .rounded))
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
            }
        }
        .frame(minHeight: 18)
    }
}

// MARK: - Day column (one bar in the chart)

private struct DayColumn: View {
    let value: Int
    let maxValue: Int
    let label: String
    let isToday: Bool
    let scheme: ColorScheme

    @State private var hovered = false

    var body: some View {
        VStack(spacing: 3) {
            Text(value > 0 ? "\(value)" : "")
                .font(.system(size: 9.5, weight: .semibold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(isToday ? JunoDesignTokens.accent : JunoTheme.secondaryText(scheme))
                .opacity(isToday || hovered ? 1 : 0)
                .frame(height: 14)

            GeometryReader { geo in
                let usable = max(0, geo.size.height)
                let pct = max(0, min(1, Double(value) / Double(max(maxValue, 1))))
                let barH = max(2, CGFloat(pct) * usable)
                VStack(spacing: 0) {
                    Spacer(minLength: 0)
                    RoundedRectangle(cornerRadius: 3, style: .continuous)
                        .fill(isToday ? JunoDesignTokens.accent : JunoUI.Calm.barRest(scheme: scheme))
                        .frame(height: barH)
                        .frame(maxWidth: 22)
                        .animation(JunoUI.Motion.cardReveal, value: value)
                }
                .frame(maxWidth: .infinity)
            }

            Text(label)
                .font(.system(size: 9, weight: isToday ? .semibold : .medium, design: .rounded))
                .tracking(0.4)
                .foregroundStyle(isToday ? JunoTheme.secondaryText(scheme) : JunoTheme.tertiaryText(scheme))
                .frame(height: 14)
        }
        .frame(maxWidth: .infinity)
        .contentShape(Rectangle())
        .onHover { hovered = $0 }
    }
}

// MARK: - App icon chip (resolve via NSWorkspace by app name)

private struct AppIconChip: View {
    let appName: String

    var body: some View {
        Image(nsImage: resolveIcon())
            .resizable()
            .scaledToFit()
            .frame(width: 16, height: 16)
            .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
            .help(appName)
            .accessibilityLabel(appName)
    }

    private func resolveIcon() -> NSImage {
        let ws = NSWorkspace.shared
        if let url = resolveApplicationURL(workspace: ws) {
            return ws.icon(forFile: url.path)
        }
        return NSImage(systemSymbolName: "app", accessibilityDescription: nil) ?? NSImage()
    }

    private func resolveApplicationURL(workspace ws: NSWorkspace) -> URL? {
        let trimmed = appName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.contains("."),
           let url = ws.urlForApplication(withBundleIdentifier: trimmed) {
            return url
        }
        let candidates = [
            "/Applications/\(trimmed).app",
            "/System/Applications/\(trimmed).app",
            "/System/Applications/Utilities/\(trimmed).app",
        ]
        for path in candidates {
            if let url = ws.urlForApplication(toOpen: URL(fileURLWithPath: path)) {
                return url
            }
            if FileManager.default.fileExists(atPath: path) {
                return URL(fileURLWithPath: path)
            }
        }
        return nil
    }
}
