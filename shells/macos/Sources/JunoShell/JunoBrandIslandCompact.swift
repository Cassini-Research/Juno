import AppKit
import SwiftUI

// MARK: - Compact HUD (live transcriptions disabled)
//
// Tiny matte pill rendered when `JunoUserDefaults.hudLiveTranscriptionsEnabled`
// is OFF. Same matte capsule shell + same brand-kit motion vocabulary as
// `JunoBrandIslandStack`, just collapsed to a single horizontal row with a
// waveform + comma. The full HUD is unchanged; this file is purely additive.
//
// State map mirrors the full HUD so behavior is consistent across both modes:
//  - Idle:          (not shown — overlay coordinator hides the panel)
//  - Listening:     comma + breath bars
//  - Refining:      comma w/ scan shimmer + processing dots
//  - Error:         danger-tinted shell + small warning glyph + shake
//  - Copy-ready:    icon-only copy button (taps the same controller method)
//  - Done (+N):     comma + tiny "+N" mono label, gated by hudShowDoneRowEnabled
//
// Animation triggers (wake / word beat / error shake) are copy-pasted from the
// full stack with identical timings — they don't share state with the full
// view, so flipping the toggle mid-session swaps surfaces cleanly.
struct JunoBrandIslandCompact: View {
    @ObservedObject var controller: DictationController
    @ObservedObject private var milestone = JunoMilestoneNotifier.shared
    @ObservedObject private var actionExecutor = JunoActionExecutor.shared

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    // MARK: State

    @State private var lastPartialWordCount: Int = 0
    @State private var wasErrorOrBlocked: Bool = false
    @State private var commaScale: CGFloat = 1
    @State private var shakeOffset: CGFloat = 0

    // MARK: Computed state

    private var isDictating: Bool {
        controller.state != "idle"
            || controller.copyableTranscript != nil
            || controller.transientActionHUDResult != nil
            || actionExecutor.inFlight != nil
    }

    private var islandIsError: Bool {
        controller.state.hasPrefix("blocked:") || controller.state.hasPrefix("error:")
    }

    private var showDoneRow: Bool {
        JunoUserDefaults.hudShowDoneRowEnabled && controller.transientDoneWordCount != nil
    }

    private enum BodyKind {
        case action(JunoActionHUDResult)
        /// "Saving N…" while ``JunoActionExecutor`` is mid-batch — keeps
        /// the pill visible during AppleScript work so the action doesn't
        /// look like it dropped on the floor.
        case actionWorking(JunoActionExecutor.InFlight)
        case copyReady
        case done(Int)
        case refining
        case error
        case listening
    }

    private var bodyKind: BodyKind {
        if let action = controller.transientActionHUDResult, controller.state == "idle" {
            return .action(action)
        }
        if let inFlight = actionExecutor.inFlight, controller.state == "idle" {
            return .actionWorking(inFlight)
        }
        if let t = controller.copyableTranscript, !t.isEmpty, controller.state == "idle" {
            return .copyReady
        }
        if let n = controller.transientDoneWordCount,
           controller.state != "refining",
           JunoUserDefaults.hudShowDoneRowEnabled {
            return .done(n)
        }
        if islandIsError { return .error }
        if controller.state == "refining" { return .refining }
        return .listening
    }

    // MARK: Body

    var body: some View {
        ZStack {
            if isDictating || showDoneRow {
                pill
                    .offset(x: shakeOffset)
                    .transition(.asymmetric(
                        insertion: .scale(scale: 0.96).combined(with: .opacity)
                            .animation(.spring(response: 0.32, dampingFraction: 0.82)),
                        removal: .opacity.animation(.easeOut(duration: 0.22))
                    ))
            }
        }
        .animation(JunoDesignTokens.pillSpring, value: controller.state)
        .animation(JunoDesignTokens.pillSpring, value: controller.transientDoneWordCount)
        // Milestone overlay lives on top of the pill (matches full HUD behavior).
        .overlay(alignment: .center) {
            if let variant = milestone.active {
                JunoBrandMilestoneOverlay(variant: variant)
                    .transition(.opacity.animation(.easeInOut(duration: 0.3)))
            }
        }
        // Behavior 03: word beat — engine still publishes liveDisplayTranscript
        // events when preview decoding is enabled at the engine level, but when
        // the user has the toggle off the engine emits nothing, so this onChange
        // simply stays quiet. No need to gate it here.
        .onChange(of: controller.liveDisplayTranscript) { newValue in
            let c = wordCount(newValue)
            if c > lastPartialWordCount, c > 0 { triggerWordBeat() }
            lastPartialWordCount = c
        }
        // Behavior 02: wake + behavior 06: error shake
        .onChange(of: controller.state) { newState in
            if newState == "idle" || newState == "checking_capability" {
                lastPartialWordCount = 0
            }
            if newState == "checking_capability"
                || newState == "checking_mic"
                || newState == "waiting_speech"
                || newState == "listening" {
                triggerWake()
            }
            let isErr = newState.hasPrefix("error:") || newState.hasPrefix("blocked:")
            if isErr && !wasErrorOrBlocked {
                // Fire only on FIRST entry into error/blocked territory.
                // Transitions between different wire strings that map to the
                // same typed state ("blocked:ax" -> "blocked:secure_field")
                // would otherwise each fire another shake.
                triggerErrorShake()
            }
            wasErrorOrBlocked = isErr
        }
    }

    // MARK: - Pill shell

    @ViewBuilder
    private var pill: some View {
        ZStack {
            JunoIslandBackground(danger: islandIsError)
            row
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
        }
        .frame(width: pillWidth)
        .fixedSize(horizontal: false, vertical: true)
        .animation(JunoDesignTokens.pillSpring, value: bodyKindKey)
    }

    /// Compact pill is content-driven within tight bounds. Listening / refining
    /// / error read at the same width (no jumping between primary states); copy
    /// and done modes are slightly tighter. Width matches the brand kit's
    /// "calm rectangle" feel — wide enough that the breath bars don't feel
    /// pinched, narrow enough that the HUD stays out of the way.
    private var pillWidth: CGFloat {
        switch bodyKind {
        case .listening, .refining, .error: return 168
        case .action:                       return 220
        case .actionWorking:                return 220
        case .copyReady:                    return 260
        case .done:                         return 132
        }
    }

    private var bodyKindKey: String {
        switch bodyKind {
        case .action:        return "action"
        case .actionWorking: return "action_work"
        case .copyReady:     return "copy"
        case .done:          return "done"
        case .refining:      return "ref"
        case .error:         return "err"
        case .listening:     return "live"
        }
    }

    // MARK: - Row content

    @ViewBuilder
    private var row: some View {
        switch bodyKind {
        case .listening:
            HStack(alignment: .center, spacing: 10) {
                JunoCommaMark(color: .white, scale: 0.32)
                    .frame(width: 18, height: 24)
                    .scaleEffect(commaScale)
                    .animation(JunoBrandKitMotion.commaBeat, value: commaScale)
                JunoBreathBars(active: controller.state == "listening", rms: controller.currentRMS)
                Spacer(minLength: 0)
            }
        case .refining:
            HStack(alignment: .center, spacing: 10) {
                ZStack(alignment: .topLeading) {
                    JunoCommaMark(color: .white, scale: 0.32)
                        .frame(width: 18, height: 24)
                    JunoScanShimmer()
                        .frame(width: 18, height: 18)
                        .clipShape(Circle())
                        .offset(x: 0, y: 3)
                }
                ProcessingDots()
                Spacer(minLength: 0)
            }
        case .error:
            HStack(alignment: .center, spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(Color.white.opacity(0.92))
                    .font(.system(size: 13))
                JunoBreathBars(active: false, rms: 0)
                Spacer(minLength: 0)
            }
        case .action(let action):
            HStack(alignment: .center, spacing: 9) {
                Image(systemName: action.symbolName)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(action.isFailure ? Color.orange.opacity(0.95) : Color.white.opacity(0.94))
                VStack(alignment: .leading, spacing: 1) {
                    Text(action.title)
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.92))
                        .lineLimit(1)
                    Text(action.subtitle)
                        .font(.system(size: 9.5, weight: .regular, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.55))
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }
        case .actionWorking(let inFlight):
            HStack(alignment: .center, spacing: 9) {
                ProcessingDots()
                VStack(alignment: .leading, spacing: 1) {
                    Text(compactWorkingTitle(for: inFlight))
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.92))
                        .lineLimit(1)
                    if inFlight.total > 1 {
                        Text("\(inFlight.completed)/\(inFlight.total)")
                            .font(.system(size: 9.5, weight: .regular, design: .monospaced))
                            .foregroundStyle(Color.white.opacity(0.50))
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
            }
        case .copyReady:
            HStack(alignment: .center, spacing: 8) {
                JunoCommaMark(color: .white, scale: 0.30)
                    .frame(width: 16, height: 22)
                VStack(alignment: .leading, spacing: 1) {
                    Text("Text ready")
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.88))
                        .lineLimit(1)
                    Text(controller.copyableTranscript ?? "")
                        .font(.system(size: 9.5, weight: .regular, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.55))
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
                copyChip
            }
        case .done(let n):
            HStack(alignment: .center, spacing: 8) {
                JunoCommaMark(color: .white, scale: 0.30)
                    .frame(width: 16, height: 22)
                    .scaleEffect(commaScale)
                // Audit fix V-4 / A2 (Compact parity): degraded-writer indicator.
                // The full Stack HUD shows a "Basic mode — writing engine not
                // loaded" line; compact has no room for prose, so use an
                // amber warning glyph alongside the "+N" word count.
                if controller.writerDegradedNotice {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(Color.orange.opacity(0.85))
                        .help("Basic output — writing engine not loaded")
                }
                Spacer(minLength: 0)
                Text("+\(n)")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color.white.opacity(0.55))
            }
        }
    }

    /// Compact working-pill title. Single-action batches read naturally;
    /// multi-action batches collapse to "Saving N notes…" / "Saving N
    /// actions…" so the tight pill width never overflows.
    private func compactWorkingTitle(for inFlight: JunoActionExecutor.InFlight) -> String {
        let unique = Set(inFlight.kinds)
        if inFlight.total == 1, let only = inFlight.kinds.first {
            return "Saving \(only.descriptor.displayName.lowercased())\u{2026}"
        }
        if unique.count == 1, let only = unique.first {
            return "Saving \(inFlight.total) \(only.descriptor.pluralName.lowercased())\u{2026}"
        }
        return "Saving \(inFlight.total) actions\u{2026}"
    }

    /// Icon-only copy button. Taps call the same controller method as the full
    /// HUD's copy button so paste/copy behavior is identical across modes.
    private var copyChip: some View {
        Button {
            controller.copyCopyableTranscriptToClipboard()
        } label: {
            Image(systemName: controller.transientCopyToast == nil ? "doc.on.doc.fill" : "checkmark")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(
                    Capsule().fill(
                        controller.transientCopyToast == nil
                            ? JunoDesignTokens.accent.opacity(0.55)
                            : JunoDesignTokens.meadow.opacity(0.82)
                    )
                )
        }
        .buttonStyle(.plain)
        .help("Copy to clipboard")
    }

    // MARK: - Animation triggers
    //
    // Copy-pasted from JunoBrandIslandStack with identical timings so the
    // brand-kit motion vocabulary stays a single source of truth across modes.

    /// Behavior 02 — Wake: scale(0.92) 80ms ease-in → scale(1.16→1.0) 270ms spring
    private func triggerWake() {
        commaScale = 0.92
        DispatchQueue.main.asyncAfter(deadline: .now() + JunoBrandKitMotion.wakeCompressDuration) {
            withAnimation(JunoBrandKitMotion.wakeExpand) {
                commaScale = 1.16
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.20) {
                withAnimation(.easeOut(duration: 0.09)) {
                    commaScale = 1.0
                }
            }
        }
    }

    /// Behavior 03 — Word beat: 1.0→1.12 (140ms) →0.97 (80ms) →1.0 (80ms)
    private func triggerWordBeat() {
        withAnimation(JunoBrandKitMotion.wordBeatIn) {
            commaScale = 1.12
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.14) {
            withAnimation(JunoBrandKitMotion.wordBeatRebound) {
                commaScale = 0.97
            }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.22) {
            withAnimation(JunoBrandKitMotion.wordBeatSettle) {
                commaScale = 1.0
            }
        }
    }

    /// Behavior 06 — Error shake: 0→-3→3→-2→2→0 in 320ms
    private func triggerErrorShake() {
        let steps: [(CGFloat, Double)] = [
            (-3, 0.055), (3, 0.055), (-2.5, 0.055),
            (2.5, 0.055), (-1.5, 0.055), (0, 0.065),
        ]
        var t = 0.0
        for (offset, dur) in steps {
            DispatchQueue.main.asyncAfter(deadline: .now() + t) {
                withAnimation(.linear(duration: dur)) { shakeOffset = offset }
            }
            t += dur
        }
    }

    // MARK: - Helpers

    private func wordCount(_ s: String) -> Int {
        s.split { $0.isWhitespace || $0.isNewline }.filter { !$0.isEmpty }.count
    }
}
