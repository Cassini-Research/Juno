// JunoHomeRecentList.swift
//
// Home's "Recent dictations" — tight one-line stream pattern (Pitch A).
// Each row is strictly one line, ellipsised, fixed 32pt height. Removes
// the ragged column edges that the multi-line clamp produced when one
// row was a single sentence and the next was three.
//
// Layout per row (4-col grid; collapses gracefully when window narrows):
//
//   [icon 18pt] [app tag auto≤80] [transcript 1fr ellipsis] [time auto] [warn-dot?]
//
// Edge-case behavior — explicit so a half-loaded broker / unknown app /
// blank transcript never breaks the page:
//
//   • 0 entries           → quiet single-line empty state.
//   • 1–4 entries         → tight rows, fixed height; section grows /
//                           shrinks naturally without reflowing siblings.
//   • Unknown app name    → upstream fallback "Unknown" (BrokerModels);
//                           we still resolve a generic app icon via SF.
//   • Missing bundle id   → SF Symbol "app" instead of NSWorkspace icon.
//   • Long app name       → truncated at 80pt, full name on hover.
//   • Empty transcript    → use whatever `historyPrimaryLine` resolves to
//                           (already falls back to "Empty dictation").
//   • Failed dictation    → amber dot next to the timestamp; no extra
//                           line so row height stays uniform.
//   • Window resized narrow → transcript flexes down; time pinned right;
//                             icon + app + dot never wrap.

import AppKit
import SwiftUI

struct JunoHomeRecentList: View {
    let entries: [UtteranceHistoryEntry]
    let onOpen: (String) -> Void
    let onSeeAll: () -> Void

    @State private var hoveredID: String?
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding(.horizontal, JunoTheme.PageInsets.detail)
                .padding(.top, 14)
                .padding(.bottom, 6)
            if entries.isEmpty {
                emptyState
                    .padding(.horizontal, JunoTheme.PageInsets.detail)
                    .padding(.vertical, 14)
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(entries.enumerated()), id: \.element.id) { index, entry in
                        Button {
                            onOpen(entry.utteranceId)
                        } label: {
                            Self.row(entry: entry, scheme: scheme, hovered: hoveredID == entry.utteranceId)
                                .contentShape(Rectangle())
                        }
                            .buttonStyle(.plain)
                            .onHover { hovering in
                                if hovering {
                                    hoveredID = entry.utteranceId
                                } else if hoveredID == entry.utteranceId {
                                    hoveredID = nil
                                }
                            }
                            .accessibilityElement(children: .combine)
                            .accessibilityLabel(Self.accessibilityLabel(for: entry))
                            .accessibilityHint("Opens this dictation in History.")
                        if index < entries.count - 1 {
                            JunoHairlineRule(.faint)
                                .padding(.leading, JunoTheme.PageInsets.detail + 18 + 10) // align under transcript col
                                .padding(.trailing, JunoTheme.PageInsets.detail)
                        }
                    }
                }
            }
        }
        .padding(.bottom, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Recent dictations")
                .junoType(.eyebrow)
                .foregroundStyle(JunoTheme.tertiaryText(scheme))
            Spacer(minLength: 8)
            Button(action: onSeeAll) {
                HStack(spacing: 3) {
                    Text("See all in History")
                    Image(systemName: "arrow.right")
                        .font(.system(size: 9, weight: .semibold))
                }
                .font(.system(size: 10.5, weight: .medium, design: .rounded))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
            }
            .buttonStyle(.plain)
            .focusable(false)
            .help("Open the History tab")
        }
    }

    private var emptyState: some View {
        // Single quiet line — no action rail. Keeps the section small
        // when there's nothing to show, and stays calm during the brief
        // window between launch and the first broker fetch.
        Text("Your last few dictations will appear here.")
            .font(.system(size: 11.5, design: .rounded))
            .foregroundStyle(JunoTheme.tertiaryText(scheme))
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Tight row

    @ViewBuilder
    private static func row(
        entry: UtteranceHistoryEntry,
        scheme: ColorScheme,
        hovered: Bool
    ) -> some View {
        HStack(spacing: 10) {
            appIcon(entry: entry)
            Text(entry.displayAppName)
                .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                .foregroundStyle(JunoTheme.tertiaryText(scheme))
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(minWidth: 50, idealWidth: 70, maxWidth: 90, alignment: .leading)
                .help(entry.displayAppName)
            Text(entry.historyPrimaryLine)
                .font(.system(size: 11.5, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
            HStack(spacing: 5) {
                Text(entry.historyTimestampLabel)
                    .font(.system(size: 9.5, weight: .medium, design: .monospaced))
                    .monospacedDigit()
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
                if hasFailureMark(entry) {
                    Circle()
                        .fill(JunoUI.Calm.amber)
                        .frame(width: 5, height: 5)
                        .help(entry.displayFailureReason ?? "Needs attention")
                }
            }
            .layoutPriority(1)  // never let the time column be eaten
        }
        .padding(.horizontal, JunoTheme.PageInsets.detail - 6)
        .padding(.vertical, 0)
        .frame(height: 32)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(hovered ? JunoUI.hairline(.faint, scheme: scheme) : Color.clear)
                .padding(.horizontal, JunoTheme.PageInsets.detail - 10)
        )
    }

    /// Quiet amber failure dot only when the entry is a dictation that
    /// actually failed — action rows surface their own status elsewhere
    /// and noisy red dots on every action history row would be wrong.
    private static func hasFailureMark(_ entry: UtteranceHistoryEntry) -> Bool {
        guard let reason = entry.failureReason, !reason.isEmpty else { return false }
        return !entry.isActionHistoryRow
    }

    private static func accessibilityLabel(for entry: UtteranceHistoryEntry) -> String {
        var parts: [String] = [entry.displayAppName, entry.historyPrimaryLine, entry.historyTimestampLabel]
        if let reason = entry.displayFailureReason, !reason.isEmpty, !entry.isActionHistoryRow {
            parts.append("Needs attention: \(reason)")
        }
        return parts.joined(separator: ", ")
    }

    /// Real macOS app icon, resolved by bundle id. Falls back to an SF
    /// Symbol so we never render a missing-image box for unknown apps.
    private static func appIcon(entry: UtteranceHistoryEntry) -> some View {
        let ws = NSWorkspace.shared
        let bundleId = entry.context?.appBundleId
        let url: URL? = bundleId.flatMap { ws.urlForApplication(withBundleIdentifier: $0) }
        let img: NSImage = {
            if let url { return ws.icon(forFile: url.path) }
            return NSImage(systemSymbolName: "app", accessibilityDescription: nil) ?? NSImage()
        }()
        return Image(nsImage: img)
            .resizable()
            .scaledToFit()
            .frame(width: 18, height: 18)
            .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
            .accessibilityHidden(true)
    }
}
