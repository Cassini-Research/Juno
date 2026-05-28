// JunoActionsHomeCard.swift
//
// Unified Home priority card for the Voice Actions feature. Replaces
// the old `JunoRemindersNudgeCard` (Reminders-only, didn't know about
// Notes or Alarms).
//
// One card, four priority states (highest first):
//   1. **Fix this** — the user just spoke an action and at least one
//      kind was blocked on permission. Surfaces the parsed body and a
//      one-tap path to the relevant Allow flow.
//   2. **Unlock** — at least one action is in `notDetermined` and the
//      user hasn't dismissed in the last 7 days.
//   3. **Try this** — all granted: a rotating tip / example so the user
//      keeps discovering what voice actions can do.
//   4. **Hidden** — Voice Actions toggle off and no failures pending.
//
// The card never blocks the page; everything dismisses cleanly with a
// 7-day cooldown per kind.

import AppKit
import Combine
import SwiftUI

extension JunoUserDefaults {
    /// 7-day cooldown timestamps per action kind, keyed by raw value.
    static func actionsNudgeDismissedAt(for kind: JunoActionKind) -> Date? {
        let key = "JunoActionsNudgeDismissedAt_\(kind.rawValue)"
        let raw = UserDefaults.standard.double(forKey: key)
        return raw > 0 ? Date(timeIntervalSince1970: raw) : nil
    }

    static func markActionsNudgeDismissed(for kind: JunoActionKind) {
        let key = "JunoActionsNudgeDismissedAt_\(kind.rawValue)"
        UserDefaults.standard.set(Date().timeIntervalSince1970, forKey: key)
    }

    static func clearAllActionsNudgeDismissals() {
        for kind in JunoActionKind.allCases {
            let key = "JunoActionsNudgeDismissedAt_\(kind.rawValue)"
            UserDefaults.standard.removeObject(forKey: key)
        }
    }

    static func dismissedRecently(for kind: JunoActionKind) -> Bool {
        guard let at = actionsNudgeDismissedAt(for: kind) else { return false }
        return Date().timeIntervalSince(at) < 7 * 24 * 3600
    }
}

struct JunoActionsHomeCard: View {
    let onTryExample: () -> Void

    @StateObject private var perms = JunoActionPermissionStore.shared
    @ObservedObject private var executor = JunoActionExecutor.shared
    @ObservedObject private var windowNav = JunoMainWindowNavigator.shared
    @AppStorage(JunoUserDefaults.actionsEnabledKey) private var actionsEnabled: Bool = false
    @Environment(\.colorScheme) private var scheme

    @State private var rotatingTipIndex: Int = 0
    @State private var rotationTask: Task<Void, Never>?

    init(onTryExample: @escaping () -> Void) {
        self.onTryExample = onTryExample
    }

    var body: some View {
        Group {
            switch state {
            case .hidden:
                // No content AND no trailing rule — when there's no cue,
                // the surrounding sections close up cleanly without a
                // visible double-divider seam.
                EmptyView()
            case .fixThis(let kind, let preview):
                cueWithDivider {
                    fixThisCard(kind: kind, preview: preview)
                }
            case .enableNudge:
                cueWithDivider {
                    enableNudgeCard()
                }
                .onAppear { markEnableNudgeShownIfNeeded() }
            case .unlock(let kind):
                cueWithDivider {
                    unlockCard(kind: kind)
                }
            case .tryThis(let descriptor, let example):
                cueWithDivider {
                    tryThisCard(descriptor: descriptor, example: example)
                }
            }
        }
        .onAppear {
            perms.beginObserving()
            updateRotationTask(for: state)
        }
        .onDisappear {
            perms.endObserving()
            stopRotationTask()
        }
        .onChange(of: state) { _, newState in
            updateRotationTask(for: newState)
        }
    }

    /// Wrap one of the three cue states with consistent page padding,
    /// stable minimum height (so swapping between states doesn't reflow),
    /// and a trailing hairline rule. The rule lives here, not in the
    /// caller, so it disappears together with the cue when it's hidden.
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

    // MARK: - State machine

    private enum CardState: Equatable {
        case hidden
        case fixThis(JunoActionKind, String)
        /// Gentle post-onboarding nudge: the user deferred Voice Actions
        /// during onboarding ("Maybe later"), and has now completed at
        /// least three successful dictations. Shown exactly once.
        case enableNudge
        case unlock(JunoActionKind)
        case tryThis(JunoActionDescriptor, String)

        static func == (lhs: CardState, rhs: CardState) -> Bool {
            switch (lhs, rhs) {
            case (.hidden, .hidden): return true
            case let (.fixThis(a, b), .fixThis(c, d)): return a == c && b == d
            case (.enableNudge, .enableNudge): return true
            case let (.unlock(a), .unlock(b)): return a == b
            case let (.tryThis(a, b), .tryThis(c, d)): return a.kind == c.kind && b == d
            default: return false
            }
        }
    }

    private func updateRotationTask(for state: CardState) {
        guard case .tryThis = state else {
            stopRotationTask()
            return
        }
        guard rotationTask == nil else { return }
        rotationTask = Task { @MainActor in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 8_000_000_000)
                guard !Task.isCancelled else { break }
                withAnimation(.easeInOut(duration: 0.4)) {
                    rotatingTipIndex &+= 1
                }
            }
        }
    }

    private func stopRotationTask() {
        rotationTask?.cancel()
        rotationTask = nil
    }

    private var state: CardState {
        // 1. Fix-this: the executor flagged a denied action this session.
        if executor.pendingPermissionAttempt,
           let preview = executor.lastUnfulfilledPreview,
           let kind = recentDeniedKind() {
            return .fixThis(kind, preview)
        }

        // 2. Post-onboarding "enable Voice Actions" nudge. Fires exactly
        // once after the user has completed onboarding (deferring Voice
        // Actions with "Maybe later") AND racked up at least three
        // successful dictations. The shown flag is set the moment the
        // card surfaces — see ``markEnableNudgeShownIfNeeded`` — so a
        // flap on relaunch can't show it twice.
        if shouldShowEnableNudge {
            return .enableNudge
        }

        // 3. Unlock: any permission that still needs user action AND
        // not dismissed. Keep this aligned with the Actions page so
        // restricted/denied states are not silent on Home.
        for descriptor in JunoActionCatalogAll {
            let status = perms.status(for: descriptor.permission)
            if status.needsUserAction,
               !JunoUserDefaults.dismissedRecently(for: descriptor.kind) {
                return .unlock(descriptor.kind)
            }
        }

        // 4. Try-this: only when actions are enabled and all granted.
        let allGranted = JunoActionCatalogAll.allSatisfy {
            perms.status(for: $0.permission).isGranted
        }
        if actionsEnabled && allGranted {
            let allExamples: [(JunoActionDescriptor, String)] = JunoActionCatalogAll.flatMap { d in
                d.examples.map { (d, $0) }
            }
            if !allExamples.isEmpty {
                let pick = allExamples[rotatingTipIndex % allExamples.count]
                return .tryThis(pick.0, pick.1)
            }
        }
        return .hidden
    }

    /// Conditions for the one-time Voice Actions enable nudge. All of:
    ///  - onboarding is completed (so we don't double-up with the
    ///    onboarding step itself)
    ///  - Voice Actions is still off (user deferred during onboarding,
    ///    or onboarding ran before this step existed)
    ///  - the nudge has never been shown
    ///  - user has completed at least 3 successful dictations
    private var shouldShowEnableNudge: Bool {
        guard JunoUserDefaults.onboardingCompleted else { return false }
        guard !actionsEnabled else { return false }
        guard !JunoUserDefaults.actionsNudgeShown else { return false }
        return JunoUserDefaults.dictationCompletedCount >= 3
    }

    /// Best-guess action kind matching the most recent denied result.
    private func recentDeniedKind() -> JunoActionKind? {
        guard let batch = executor.recentBatch else { return nil }
        return batch.results.first {
            $0.status == .permissionDenied || $0.status == .blockedNoPermission
        }?.kind
    }

    // MARK: - Sub-cards (cue-card re-skin)
    //
    // All three states render through the same slim row layout so
    // navigating between them feels like one card breathing, not three
    // separate panels. No `.premiumCard()` here — Home composes sections
    // with hairline rules; the cue is a strip, not a panel.

    private func fixThisCard(kind: JunoActionKind, preview: String) -> some View {
        let descriptor = kind.descriptor
        return cueRow(
            label: "Last action didn't land",
            primary: "“\(String(preview.prefix(80)))”",
            ghost: " — that \(descriptor.displayName.lowercased()) didn't save.",
            cta: "Fix in Actions",
            ctaAction: {
                windowNav.section = .actions
                executor.clearPendingPermissionAttempt()
            },
            dismiss: {
                executor.clearPendingPermissionAttempt()
            },
            showRotator: false
        )
    }

    /// One-time post-onboarding nudge. Tapping the CTA navigates to the
    /// Actions page (where the user can flip the master toggle and grant
    /// permissions). Tapping the dismiss x records it as never-show-again.
    /// Either way, the "shown" flag was already set on first appear so the
    /// nudge can never repeat itself.
    private func enableNudgeCard() -> some View {
        cueRow(
            label: "Try saying",
            primary: "\u{201C}Juno, take a note about today\u{201D}",
            ghost: " — voice actions are off, but you can turn them on.",
            cta: "Turn on",
            ctaAction: {
                // Deep-link into Settings → Actions. The Actions sidebar
                // entry is where the master toggle and permission Allow
                // buttons live, so it's the natural landing spot. Mark
                // the nudge as never-show-again on click.
                JunoUserDefaults.actionsNudgeShown = true
                windowNav.section = .actions
            },
            dismiss: {
                JunoUserDefaults.actionsNudgeShown = true
            },
            showRotator: false
        )
    }

    /// Mark the nudge as "shown" the moment it first appears on screen,
    /// so a window-close / relaunch can't surface it a second time. The
    /// dismiss x and the CTA both also set the flag for safety, but
    /// this is the load-bearing call.
    private func markEnableNudgeShownIfNeeded() {
        if !JunoUserDefaults.actionsNudgeShown {
            JunoUserDefaults.actionsNudgeShown = true
        }
    }

    private func unlockCard(kind: JunoActionKind) -> some View {
        let descriptor = kind.descriptor
        return cueRow(
            label: "One-time setup",
            primary: "Talk \(descriptor.displayName.lowercased())s into existence",
            ghost: " — \(descriptor.blurb)",
            cta: "Set up",
            ctaAction: { windowNav.section = .actions },
            dismiss: { JunoUserDefaults.markActionsNudgeDismissed(for: kind) },
            showRotator: false
        )
    }

    private func tryThisCard(descriptor: JunoActionDescriptor, example: String) -> some View {
        cueRow(
            label: "Try saying",
            primary: "“\(example)”",
            ghost: nil,
            cta: "Try it",
            ctaAction: onTryExample,
            dismiss: nil,
            showRotator: true
        )
    }

    /// Shared slim cue row — quote glyph, eyebrow + utterance, dark CTA pill,
    /// dismiss x, rotator dots. Used by all three cue states.
    @ViewBuilder
    private func cueRow(
        label: String,
        primary: String,
        ghost: String?,
        cta: String,
        ctaAction: @escaping () -> Void,
        dismiss: (() -> Void)?,
        showRotator: Bool
    ) -> some View {
        ZStack(alignment: .topTrailing) {
            HStack(alignment: .center, spacing: 12) {
                Text("\u{201C}")
                    .font(.system(size: 22, design: .serif))
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
                    .opacity(0.6)
                    .padding(.top, 8)
                    .frame(width: 12, alignment: .center)

                VStack(alignment: .leading, spacing: 2) {
                    Text(label.uppercased())
                        .junoType(.eyebrow)
                        .foregroundStyle(JunoTheme.tertiaryText(scheme))
                    cueUtterance(primary: primary, ghost: ghost)
                        .lineLimit(2)
                }

                Spacer(minLength: 8)

                Button(cta, action: ctaAction)
                    .buttonStyle(JunoCueCTAButtonStyle())
                    .focusable(false)

                if let dismiss {
                    Button(action: dismiss) {
                        Image(systemName: "xmark")
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .buttonStyle(.plain)
                    .focusable(false)
                    .foregroundStyle(JunoTheme.tertiaryText(scheme))
                    .frame(width: 18, height: 18)
                    .help("Hide for a week")
                }
            }
            if showRotator {
                rotatorDots
                    .padding(.top, 0)
                    .padding(.trailing, 0)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Render the utterance with an optional dimmed ghost continuation.
    @ViewBuilder
    private func cueUtterance(primary: String, ghost: String?) -> some View {
        if let ghost {
            (Text(primary)
                .font(.system(size: 12.5, weight: .medium, design: .rounded))
                .foregroundColor(JunoTheme.primaryText(scheme))
            +
            Text(ghost)
                .font(.system(size: 12.5, weight: .regular, design: .rounded))
                .foregroundColor(JunoTheme.tertiaryText(scheme)))
        } else {
            Text(primary)
                .font(.system(size: 12.5, weight: .medium, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
        }
    }

    /// Three little 8×2 capsules at top-right that visualize the 8-second
    /// rotation cadence on the Try-this state. Purely decorative — the
    /// timer in `body.onReceive` is the source of truth.
    private var rotatorDots: some View {
        HStack(spacing: 3) {
            ForEach(0..<3, id: \.self) { i in
                Capsule(style: .continuous)
                    .fill(
                        i == (rotatingTipIndex % 3)
                            ? JunoTheme.tertiaryText(scheme)
                            : JunoUI.hairline(.regular, scheme: scheme)
                    )
                    .frame(width: 8, height: 2)
            }
        }
    }
}

/// Slim navy CTA pill — 999 radius, white text, tight padding. Used only
/// by the cue card. Distinct from `JunoPrimaryActionButtonStyle` (which is
/// used for full-width primary actions on settings/actions pages).
private struct JunoCueCTAButtonStyle: ButtonStyle {
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
