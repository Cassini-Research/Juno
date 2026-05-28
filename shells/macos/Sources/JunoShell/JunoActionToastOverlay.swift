// JunoActionToastOverlay.swift
//
// Global toast that surfaces the result of every Voice Action batch,
// regardless of which page the user is currently viewing. Replaces the
// Home-only chip stack so users on History, Settings, etc. still get
// confirmation that their "take a note" / "remind me at 5pm" landed.
//
// Auto-dismisses after a few seconds. The user can also dismiss
// manually. Tapping a chip's "Open" button opens the saved note or
// reminder.

import SwiftUI

struct JunoActionToastOverlay: View {

    @ObservedObject private var executor = JunoActionExecutor.shared
    @ObservedObject private var windowNav = JunoMainWindowNavigator.shared
    @State private var dismissedBatchId: String?
    @State private var dismissTask: DispatchWorkItem?

    var body: some View {
        Group {
            if let batch = executor.recentBatch,
               batch.utteranceId != dismissedBatchId
            {
                JunoActionToastCard(batch: batch, onDismiss: { dismiss(batch.utteranceId) }) {
                    handleAction(for: batch)
                }
                .frame(maxWidth: 430)
                .transition(.move(edge: .trailing).combined(with: .opacity))
                .onAppear { scheduleDismiss(batch) }
                .onChange(of: batch.utteranceId) { newId in
                    scheduleDismiss(batch)
                }
            }
        }
        .animation(.easeOut(duration: 0.22), value: executor.recentBatch?.utteranceId)
    }

    private func scheduleDismiss(_ batch: JunoActionExecutor.ActionBatch) {
        dismissTask?.cancel()
        let task = DispatchWorkItem { dismiss(batch.utteranceId) }
        dismissTask = task
        let delay = min(14.0, max(6.0, Double(batch.results.count) * 0.45))
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: task)
    }

    private func dismiss(_ id: String) {
        dismissedBatchId = id
        executor.clearRecent()
    }

    private func handleAction(for batch: JunoActionExecutor.ActionBatch) {
        // Blocked-by-toggle: jump to Actions page so user can flip it on.
        if batch.results.contains(where: { $0.status == .blockedToggleOff }) {
            windowNav.section = .actions
            dismiss(batch.utteranceId)
            return
        }
        // Missing permission: same — Actions page owns the grant flow.
        if batch.results.contains(where: { $0.status == .blockedNoPermission || $0.status == .permissionDenied }) {
            windowNav.section = .actions
            dismiss(batch.utteranceId)
            return
        }
    }
}

// MARK: - Card

private struct JunoActionToastCard: View {
    let batch: JunoActionExecutor.ActionBatch
    let onDismiss: () -> Void
    let onResolveAction: () -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            // One-line summary visible above the per-row stack — lets
            // the user grok partial success ("Saved 2 of 3.") at a
            // glance even before reading the chip stack. Suppressed for
            // single-action batches where the headline already says
            // everything.
            if let detail = summary.detail, batch.results.count > 1 {
                Text(detail)
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Divider().opacity(0.4)
            ScrollView(.vertical, showsIndicators: batch.results.count > 6) {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(zip(batch.requests, batch.results).enumerated()), id: \.offset) { _, pair in
                        row(request: pair.0, result: pair.1)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 360)
            if needsResolveButton {
                Button(action: onResolveAction) {
                    HStack(spacing: 4) {
                        Text(resolveButtonLabel)
                            .font(.system(size: 11.5, weight: .semibold, design: .rounded))
                        Image(systemName: "arrow.right")
                            .font(.system(size: 9, weight: .bold))
                    }
                }
                .buttonStyle(.plain)
                .foregroundStyle(JunoDesignTokens.accent)
                .padding(.top, 2)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme))
                .shadow(color: Color.black.opacity(scheme == .dark ? 0.35 : 0.12), radius: 18, y: 8)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme).opacity(scheme == .dark ? 0.5 : 0.18), lineWidth: 0.6)
        )
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: headlineIcon)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(headlineColor)
            Text(headline)
                .junoType(.bodyEmphasis)
                .foregroundStyle(JunoTheme.primaryText(scheme))
            Spacer(minLength: 0)
            Button(action: onDismiss) {
                Image(systemName: "xmark")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .padding(4)
            }
            .buttonStyle(.plain)
            .help("Dismiss")
        }
    }

    private func row(request: JunoActionRequest, result: JunoActionResult) -> some View {
        // Multi-row batches collapse each chip to one tight line so 5+
        // long-bodied notes don't render as a wall of text. Single-row
        // batches keep the original two-line breathing room because
        // there's nothing competing for height.
        let dense = batch.results.count > 1
        return HStack(alignment: .top, spacing: 10) {
            Image(systemName: request.kind.descriptor.symbolName)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(request.kind.descriptor.accent)
                .frame(width: 18)
                .padding(.top, 1)

            VStack(alignment: .leading, spacing: 2) {
                Text(rowTitle(request: request, result: result, dense: dense))
                    .junoType(.label)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .lineLimit(dense ? 1 : 2)
                    .truncationMode(.tail)
                if let secondary = rowSecondary(request: request, result: result) {
                    Text(secondary)
                        .junoType(.caption)
                        .foregroundStyle(secondaryColor(for: request, result: result))
                        .lineLimit(dense ? 1 : 2)
                        .truncationMode(.tail)
                }
            }

            Spacer(minLength: 0)

            if let url = sinkURL(for: result) {
                Button("Open") { NSWorkspace.shared.open(url) }
                    .buttonStyle(.plain)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(JunoDesignTokens.accent)
            } else {
                statusBadge(for: result)
            }
        }
    }

    // MARK: Headline (shared formatter — see ``JunoActionBatchFormatter``)

    private var summary: JunoActionBatchSummary {
        JunoActionBatchFormatter.summarize(batch.results)
    }

    private var headline: String { summary.headline }

    private var headlineIcon: String {
        switch summary.tone {
        case .allSaved: return "checkmark.circle.fill"
        case .partial: return "checkmark.circle.badge.questionmark"
        case .blocked:
            return batch.results.contains(where: { $0.status == .blockedToggleOff })
                ? "switch.2" : "lock.fill"
        case .failed: return "exclamationmark.triangle.fill"
        case .allPending: return "arrow.triangle.2.circlepath"
        }
    }

    private var headlineColor: Color {
        switch summary.tone {
        case .allSaved: return .green
        case .partial: return JunoDesignTokens.accent
        case .blocked: return .orange
        case .failed: return .red
        case .allPending: return JunoDesignTokens.accent
        }
    }

    // MARK: Row helpers

    private func rowTitle(request: JunoActionRequest, result: JunoActionResult, dense: Bool = false) -> String {
        let rawBody = result.bodyPreview.isEmpty ? request.body : result.bodyPreview
        let kindWord = request.kind.descriptor.displayName
        // Tighten body previews in dense (multi-row) layouts so each chip
        // stays one line. Trailing ellipsis is added by SwiftUI's
        // truncationMode(.tail), so we just clamp the source string here
        // to keep the layout pass cheap and predictable.
        let limit = dense ? 56 : 80
        let body = rawBody.count > limit
            ? String(rawBody.prefix(limit)).trimmingCharacters(in: .whitespacesAndNewlines) + "\u{2026}"
            : rawBody
        return body.isEmpty ? kindWord : "\(kindWord): \(body)"
    }

    private func rowSecondary(request: JunoActionRequest, result: JunoActionResult) -> String? {
        switch result.status {
        case .ok:
            // Recurrence preview (Phase 1): when the action carries a
            // schedule.series, render the rule in plain English ("Daily
            // for 10 days at 9 AM") instead of just the first-fire time.
            // The first-fire-only rendering hid that the action was
            // recurring at all.
            if let series = request.schedule?.series,
               let summary = JunoRecurrenceCopy.summary(for: series) {
                let tail = request.kind == .alarm ? " · Calendar alert" : ""
                return summary + tail
            }
            if let vague = request.schedule?.vague,
               let formatted = formatDue(vague.defaultIso) {
                return "\(formatted) · tap to change"
            }
            // For reminders/alarms, prefer the time. For notes (no time),
            // show the destination ("Juno folder · Apple Notes") so the
            // user learns where to look. Calendar-event alarms also get
            // a "Calendar alert" tail so the messaging stays consistent
            // with the action card and the HUD subtitle.
            if let when = request.when, let formatted = formatDue(when.iso) {
                let tail: String
                switch request.kind {
                case .alarm: tail = " · Calendar alert"
                default: tail = ""
                }
                let prefix = when.inferred ? "\(formatted) · time inferred" : formatted
                return prefix + tail
            }
            switch request.kind {
            case .note:
                return "\(JunoNotesFolderName) folder · Apple Notes"
            case .alarm:
                return "Calendar alert"
            case .reminder:
                return "Apple Reminders"
            }
        case .blockedToggleOff:
            return "Tap to enable in Actions."
        case .blockedNoPermission, .permissionDenied:
            return result.error ?? "Tap to grant access."
        case .sinkError, .timeParseFailed:
            return result.error
        case .pending:
            // Matches the History UI. The action almost always *did*
            // run; we just haven't received the result post yet.
            return "Saving \u{2026}"
        }
    }

    private func secondaryColor(for request: JunoActionRequest, result: JunoActionResult) -> Color {
        switch result.status {
        case .ok:
            return request.schedule?.vague?.needsConfirmation == true ? .orange : JunoTheme.secondaryText(scheme)
        case .blockedToggleOff: return JunoTheme.secondaryText(scheme)
        case .blockedNoPermission, .permissionDenied: return .orange
        case .sinkError, .timeParseFailed: return .red
        case .pending: return JunoTheme.secondaryText(scheme)
        }
    }

    @ViewBuilder
    private func statusBadge(for result: JunoActionResult) -> some View {
        switch result.status {
        case .ok:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .permissionDenied, .blockedNoPermission:
            Image(systemName: "lock.fill").foregroundStyle(.orange)
        case .sinkError, .timeParseFailed:
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
        case .blockedToggleOff:
            Image(systemName: "switch.2").foregroundStyle(.secondary)
        case .pending:
            Image(systemName: "circle.dashed").foregroundStyle(.secondary)
        }
    }

    private func sinkURL(for result: JunoActionResult) -> URL? {
        guard result.status == .ok, let s = result.sinkUrl else { return nil }
        return URL(string: s)
    }

    private var needsResolveButton: Bool { summary.needsResolveCTA }

    private var resolveButtonLabel: String {
        if batch.results.contains(where: { $0.status == .blockedToggleOff }) {
            return "Open Voice Actions"
        }
        // Partial success shouldn't say "Grant access" — it sounds like
        // nothing saved. Lean on copy that matches what the headline
        // already conveyed ("Saved 4 of 5").
        if summary.tone == .partial {
            return "Finish setup"
        }
        return "Grant access"
    }

    private static let dueFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "EEE, MMM d 'at' h:mm a"
        return f
    }()

    private func formatDue(_ iso: String) -> String? {
        let p = ISO8601DateFormatter()
        p.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = p.date(from: iso) { return Self.dueFormatter.string(from: d) }
        p.formatOptions = [.withInternetDateTime]
        if let d = p.date(from: iso) { return Self.dueFormatter.string(from: d) }
        return nil
    }
}
