import AppKit
import SwiftUI

// MARK: - Floating HUD (all dictation states + milestone overlay)

/// Single-shell HUD: one matte capsule whose internal content morphs between
/// pre-listening / listening / refining / done / copy-ready / error states.
/// The size is content-driven within tight bounds so transitions read as
/// "the same surface" rather than the pill jumping shape.
///
/// Behavior map:
///  01 — Idle breathe   (not shown; panel hidden when idle)
///  02 — Wake           triggered on first listening/checking_capability
///  03 — Word beat      triggered per new word in the engine-owned HUD transcript
///  04 — Scan shimmer   shown during refining/processing state
///  05 — Draft flash    ring pulse on the island after text is placed
///  06 — Error shake    triggered when state starts with "error:" or "blocked:"
///  07/08 — Milestone   separate JunoBrandMilestoneOverlay overlay
struct JunoBrandIslandStack: View {
    @ObservedObject var controller: DictationController
    @ObservedObject private var milestone = JunoMilestoneNotifier.shared
    @ObservedObject private var actionExecutor = JunoActionExecutor.shared

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    // MARK: State

    @State private var lastPartialWordCount: Int = 0
    @State private var wasErrorOrBlocked: Bool = false
    @State private var lastWordBeatAt: Date = .distantPast
    @State private var lastTranscriptScrollAt: Date = .distantPast
    @State private var lastTranscriptScrollWordCount: Int = 0
    /// Comma mark scale — driven by wake (02) and word beat (03)
    @State private var commaScale: CGFloat = 1
    /// Horizontal shake offset — driven by error shake (06)
    @State private var shakeOffset: CGFloat = 0
    /// Commands reference popover — shown in copy-ready state
    @State private var showCommandsPopover = false
    /// One-shot shimmer pulse on the live transcript when the engine
    /// replaces preview text with a corrected/final version. Drives the
    /// "magical replace" moment the user asked for: a single soft
    /// white glow that fades in for ~140ms then back out for ~140ms.
    /// Restrained on purpose — billions of impressions a day reward
    /// subtlety over fireworks.
    @State private var correctionPulseActive: Bool = false
    @State private var correctionPulseGeneration: Int = 0

    // MARK: Computed state

    private var isDictating: Bool {
        controller.hudState != .idle
            || controller.copyableTranscript != nil
            || controller.transientActionHUDResult != nil
            || actionExecutor.inFlight != nil
    }

    private var islandIsError: Bool {
        controller.hudState.isErrorOrBlocked
    }

    private var showDoneRow: Bool {
        JunoUserDefaults.hudShowDoneRowEnabled
            && controller.transientDoneWordCount != nil
            && controller.hudState == .idle
    }

    /// Top-level body branch: copy-ready and done are special compact rows; everything
    /// else uses the unified listening/refining/error shell.
    private enum BodyKind {
        case copyReady(String)
        case action(JunoActionHUDResult)
        /// "Saving 3 notes…" — shown while ``JunoActionExecutor`` is mid-batch.
        /// Closes the perception gap between speech ending and the success
        /// HUD appearing; previously the island faded out for the duration of
        /// the AppleScript work and the action looked like it had failed.
        case actionWorking(JunoActionExecutor.InFlight)
        case done(Int)
        case refining
        case error
        case listening
    }

    private var bodyKind: BodyKind {
        if let t = controller.copyableTranscript, !t.isEmpty, controller.hudState == .idle {
            return .copyReady(t)
        }
        if let action = controller.transientActionHUDResult, controller.hudState == .idle {
            return .action(action)
        }
        // Working state takes the slot once dictation has settled to idle —
        // the executor only runs after the broker returns, so ``state`` is
        // already idle by then. Show the working pill until the result HUD
        // arrives.
        if let inFlight = actionExecutor.inFlight, controller.hudState == .idle {
            return .actionWorking(inFlight)
        }
        if let n = controller.transientDoneWordCount,
           controller.hudState == .idle,
           JunoUserDefaults.hudShowDoneRowEnabled {
            return .done(n)
        }
        if islandIsError { return .error }
        if controller.hudState == .refining { return .refining }
        return .listening
    }

    // MARK: Body

    var body: some View {
        ZStack {
            if isDictating || showDoneRow {
                mainIsland
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
        // Milestone overlay lives on top of the island
        .overlay(alignment: .center) {
            if let variant = milestone.active {
                JunoBrandMilestoneOverlay(variant: variant)
                    .transition(.opacity.animation(.easeInOut(duration: 0.3)))
            }
        }
        // Behavior 03: word beat
        .onChange(of: controller.liveDisplayTranscript) { newValue in
            let c = wordCount(newValue)
            if c > lastPartialWordCount, c > 0 { triggerWordBeat() }
            lastPartialWordCount = c
        }
        // Behavior 02: wake + behavior 06: error shake
        .onChange(of: controller.state) { newState in
            let newHUD = HUDState.from(wireString: newState)
            if newHUD == .idle || newHUD == .checkingCapability {
                lastPartialWordCount = 0
            }
            if newHUD == .checkingCapability
                || newHUD == .checkingMic
                || newHUD == .waitingSpeech
                || newHUD == .listening {
                triggerWake()
            }
            let isErr = newHUD.isErrorOrBlocked
            if isErr && !wasErrorOrBlocked {
                // Fire only on FIRST entry into error/blocked territory. Without
                // this dedup, transitions between different blocked reasons
                // ("blocked:ax" -> "blocked:secure_field") or between different
                // error strings each fire another shake — including during the
                // engine's restart loop where state may oscillate through
                // blocked:broker_unreachable several times.
                triggerErrorShake()
            }
            wasErrorOrBlocked = isErr
        }
    }

    // MARK: - Main island shell

    @ViewBuilder
    private var mainIsland: some View {
        ZStack {
            islandBackground(danger: islandIsError)
            // Behavior 05: draft flash ring (opacity-only — scale fights the fixed shell)
            if controller.draftFlashActive {
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(Color.white.opacity(0.55), lineWidth: 1.0)
                    .opacity(controller.draftFlashActive ? 1 : 0)
                    .animation(.timingCurve(0.4, 0, 0.2, 1, duration: 0.28), value: controller.draftFlashActive)
            }
            if controller.delightSweepActive {
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(Color.white.opacity(0.24), lineWidth: 1.0)
                    .transition(.opacity)
            }

            VStack(spacing: 8) {
                topStripe
                    .frame(maxWidth: .infinity, alignment: .leading)
                bodyContent
                    .frame(maxWidth: .infinity, alignment: .leading)
                if showFooter {
                    footer
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
        }
        .frame(width: JunoDesignTokens.islandWidth)
        .fixedSize(horizontal: false, vertical: true)
        .animation(JunoDesignTokens.pillSpring, value: bodyKindKey)
    }

    /// Discriminator that fires the pill-spring on body-kind changes (which alters height).
    private var bodyKindKey: String {
        switch bodyKind {
        case .copyReady: return "copy"
        case .action(let result): return result.isFailure ? "action_err" : "action_ok"
        case .actionWorking: return "action_work"
        case .done: return "done"
        case .refining: return "ref"
        case .error: return "err"
        case .listening: return "live"
        }
    }

    // MARK: - Top stripe

    @ViewBuilder
    private var topStripe: some View {
        switch bodyKind {
        case .error:
            HStack(spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(Color.white.opacity(0.92))
                    .font(.system(size: 13))
                Text(errorTitle)
                    .font(.system(size: 12.5, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.92))
                    .lineLimit(1)
                Spacer(minLength: 0)
            }
        case .done(let n):
            HStack(spacing: 10) {
                JunoCommaMark(color: .white, scale: 0.32)
                    .frame(width: 20, height: 28)
                    .scaleEffect(commaScale)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Text placed")
                        .font(.system(size: 13, weight: .medium, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.88))
                    // Audit fix V-4 / A2: when the engine reports
                    // degraded_writer, surface a quiet basic-mode line
                    // so polish-mode users have a signal that the LLM
                    // writer didn't load.
                    if controller.writerDegradedNotice {
                        Text("Basic output — writing engine not loaded")
                            .font(.system(size: 10, weight: .regular, design: .rounded))
                            .foregroundStyle(Color.white.opacity(0.55))
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
                Text("+\(n) words")
                    .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color.white.opacity(0.40))
            }
        case .copyReady:
            HStack(spacing: 10) {
                JunoCommaMark(color: .white, scale: 0.32)
                    .frame(width: 20, height: 28)
                Text(controller.transientCopyToast ?? "Tap to copy")
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(
                        controller.transientCopyToast == nil
                            ? Color.white.opacity(0.65)
                            : JunoDesignTokens.meadow.opacity(0.95)
                    )
                    .tracking(0.4)
                Spacer(minLength: 0)
                Button {
                    showCommandsPopover.toggle()
                } label: {
                    Image(systemName: "questionmark")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(Color.white.opacity(0.52))
                        .frame(width: 18, height: 18)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Voice commands reference")
                .popover(isPresented: $showCommandsPopover, arrowEdge: .bottom) {
                    hudCommandsPopover
                }
                copyButton
            }
        case .action(let result):
            HStack(spacing: 10) {
                if let kind = result.kind, !result.isFailure {
                    JunoActionNativeIcon(kind: kind, size: 20, fallbackColor: JunoDesignTokens.meadow.opacity(0.95))
                } else {
                    Image(systemName: result.symbolName)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(result.isFailure ? Color.orange.opacity(0.95) : JunoDesignTokens.meadow.opacity(0.95))
                        .frame(width: 20)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(result.title)
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.90))
                        .lineLimit(1)
                    Text(result.subtitle)
                        .font(.system(size: 10.5, weight: .medium, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.56))
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }
        case .actionWorking(let inFlight):
            HStack(spacing: 10) {
                ProcessingDots()
                    .frame(width: 20)
                VStack(alignment: .leading, spacing: 2) {
                    Text(actionWorkingTitle(for: inFlight))
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.90))
                        .lineLimit(1)
                    Text(actionWorkingSubtitle(for: inFlight))
                        .font(.system(size: 10.5, weight: .medium, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.56))
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
                if inFlight.total > 1 {
                    Text("\(inFlight.completed)/\(inFlight.total)")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Color.white.opacity(0.45))
                        .tracking(0.6)
                }
            }
        case .refining:
            HStack(spacing: 10) {
                ZStack(alignment: .topLeading) {
                    JunoCommaMark(color: .white, scale: 0.35)
                        .frame(width: 22, height: 30)
                    JunoScanShimmer()
                        .frame(width: 22, height: 22)
                        .clipShape(Circle())
                        .offset(x: 0, y: 4)
                }
                ProcessingDots()
                Text("Transcribing")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.55))
                Spacer(minLength: 0)
                refiningElapsedView
            }
        case .listening:
            HStack(alignment: .center, spacing: 10) {
                JunoCommaMark(color: .white, scale: 0.34)
                    .frame(width: 22, height: 30)
                    .scaleEffect(commaScale)
                    .animation(JunoBrandKitMotion.commaBeat, value: commaScale)
                JunoBreathBars(active: controller.hudState == .listening, rms: controller.currentRMS)
                Spacer(minLength: 0)
                Text(stripeStatusLabel)
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Color.white.opacity(0.65))
                    .tracking(0.6)
                if controller.dictationStartedAt != nil {
                    listeningTimer
                }
                stopControl
            }
        }
    }

    private var stripeStatusLabel: String {
        switch controller.hudState {
        case .checkingCapability: return "PREPARING"
        case .checkingMic:        return "MIC"
        case .waitingSpeech:      return "WAITING"
        case .partialCommit:      return "CORRECTING"
        default:
            // Surface that local live captions are available without making
            // normal waiting states look like a broken mode.
            return "LISTENING"
        }
    }

    // MARK: - Body content

    @ViewBuilder
    private var bodyContent: some View {
        switch bodyKind {
        case .listening, .refining:
            transcriptOrPlaceholder
        case .copyReady(let t):
            copyReadyTranscript(t)
        case .action:
            EmptyView()
        case .actionWorking:
            EmptyView()
        case .done:
            EmptyView()
        case .error:
            Text(errorMessage)
                .font(.system(size: 11.5, weight: .regular, design: .rounded))
                .foregroundStyle(Color.white.opacity(0.78))
                .lineLimit(2)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - Working-state copy

    /// Title shown in the working pill. Kind-aware so single-action batches
    /// read naturally ("Saving note…") and multi-kind batches collapse to
    /// "Saving N actions…" instead of fabricating a sentence.
    private func actionWorkingTitle(for inFlight: JunoActionExecutor.InFlight) -> String {
        let unique = Set(inFlight.kinds)
        if inFlight.total == 1, let only = inFlight.kinds.first {
            return "Saving \(only.descriptor.displayName.lowercased())\u{2026}"
        }
        if unique.count == 1, let only = unique.first {
            return "Saving \(inFlight.total) \(only.descriptor.pluralName.lowercased())\u{2026}"
        }
        return "Saving \(inFlight.total) actions\u{2026}"
    }

    /// Subtitle teases the destination so the user doesn't wonder where the
    /// work is landing. Mirrors ``actionHUDSubtitle(for:)`` in JunoShellApp.
    private func actionWorkingSubtitle(for inFlight: JunoActionExecutor.InFlight) -> String {
        let unique = Set(inFlight.kinds)
        if unique.count == 1, let only = unique.first {
            switch only {
            case .note:     return "Notes \u{2192} \(JunoNotesFolderName) folder"
            case .reminder: return "Reminders"
            case .alarm:    return "Alarm"
            }
        }
        // Mixed-kind batches: list the destinations in the order the kinds
        // first appear so the user can recognize their own request order.
        var seen: [JunoActionKind] = []
        for k in inFlight.kinds where !seen.contains(k) { seen.append(k) }
        let names = seen.map { kind -> String in
            switch kind {
            case .note:     return "Notes"
            case .reminder: return "Reminders"
            case .alarm:    return "Alarm"
            }
        }
        return names.joined(separator: " · ")
    }

    @ViewBuilder
    private var transcriptOrPlaceholder: some View {
        if controller.liveDisplayTranscript.isEmpty {
            HStack(spacing: 6) {
                Text(placeholderWhileListening)
                    .font(.system(size: 12.5, weight: .medium, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.62))
                if let hint = controller.liveSpeechHint, !hint.isEmpty {
                    Text("·")
                        .foregroundStyle(Color.orange.opacity(0.65))
                        .font(.system(size: 11))
                    Text(hint)
                        .font(.system(size: 11, weight: .regular, design: .rounded))
                        .foregroundStyle(Color.orange.opacity(0.85))
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            ScrollViewReader { proxy in
                ScrollView(.vertical, showsIndicators: false) {
                    VStack(spacing: 0) {
                        liveTranscriptWords
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .multilineTextAlignment(.leading)
                        Color.clear
                            .frame(height: 1)
                            .id("live_transcript_bottom")
                    }
                }
                .frame(minHeight: 22, maxHeight: JunoDesignTokens.listeningTranscriptScrollMaxHeight)
                .onAppear {
                    proxy.scrollTo("live_transcript_bottom", anchor: .bottom)
                }
                .onChange(of: controller.liveDisplayTranscript) { _ in
                    let now = Date()
                    let words = wordCount(controller.liveDisplayTranscript)
                    let shouldScroll = words != lastTranscriptScrollWordCount
                        || now.timeIntervalSince(lastTranscriptScrollAt) >= 0.18
                    guard shouldScroll else { return }
                    lastTranscriptScrollAt = now
                    lastTranscriptScrollWordCount = words
                    withAnimation(.easeOut(duration: 0.12)) {
                        proxy.scrollTo("live_transcript_bottom", anchor: .bottom)
                    }
                }
                .mask(transcriptFadeMask)
            }
        }
    }

    @ViewBuilder
    private var liveTranscriptWords: some View {
        if controller.liveTranscriptSpans.isEmpty {
            Text(controller.liveDisplayTranscript)
                .font(.system(size: 13, weight: .medium, design: .rounded))
                .foregroundColor(Color.white.opacity(0.92))
        } else {
            JunoFlowLayout(spacing: 3.5, runSpacing: 2) {
                ForEach(controller.liveTranscriptSpans) { span in
                    LiveTranscriptWord(span: span, reduceMotion: reduceMotion)
                        .transition(
                            .asymmetric(
                                insertion: .opacity.combined(with: .move(edge: .bottom)),
                                removal: .opacity.combined(with: .move(edge: .top))
                            )
                        )
                        .animation(.easeOut(duration: 0.14), value: span.id)
                }
            }
        }
    }

    /// Soft top fade-out so scroll-in text doesn't cut harshly against the topStripe.
    private var transcriptFadeMask: some View {
        LinearGradient(
            stops: [
                .init(color: .clear, location: 0.0),
                .init(color: .black, location: 0.10),
                .init(color: .black, location: 1.0),
            ],
            startPoint: .top, endPoint: .bottom
        )
    }

    private var placeholderWhileListening: String {
        switch controller.hudState {
        case .checkingCapability, .checkingMic: return "Getting ready…"
        case .waitingSpeech:                    return "Speak when ready"
        case .partialCommit:                    return "…"
        default:                                return "Speak now"
        }
    }

    private func copyReadyTranscript(_ transcript: String) -> some View {
        ScrollView(.vertical, showsIndicators: true) {
            Text(transcript)
                .font(.system(size: 12.5, weight: .medium, design: .rounded))
                .foregroundStyle(Color.white.opacity(0.86))
                .frame(maxWidth: .infinity, alignment: .leading)
                .multilineTextAlignment(.leading)
                .textSelection(.enabled)
        }
        .frame(minHeight: 32, maxHeight: JunoDesignTokens.copyReadyTranscriptMaxHeight)
    }

    private var copyButton: some View {
        Button {
            controller.copyCopyableTranscriptToClipboard()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: controller.transientCopyToast == nil ? "doc.on.doc.fill" : "checkmark")
                    .font(.system(size: 11, weight: .semibold))
                Text(controller.transientCopyToast ?? "Copy")
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 10)
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

    private var stopControl: some View {
        Button {
            controller.toggleDictation()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "stop.fill")
                    .font(.system(size: 8.5, weight: .bold))
                Text("Stop")
                    .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                JunoKeycap(label: stopHintLabels().keycap)
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(Capsule(style: .continuous).fill(JunoDesignTokens.danger.opacity(0.82)))
            .overlay(
                Capsule(style: .continuous)
                    .strokeBorder(Color.white.opacity(0.18), lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .help("Stop dictation")
        .accessibilityLabel("Stop dictation")
    }

    // MARK: - Commands reference popover

    private var hudCommandsPopover: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "waveform.and.mic")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(JunoDesignTokens.accent)
                Text("Voice Commands")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
            }

            hudCommandGroup(title: "In the moment", commands: kHudInMomentCommands)
            hudCommandGroup(title: "Edit recent text", commands: kHudRecentEditCommands)
            hudCommandGroup(title: "When text is selected", commands: kHudSelectionCommands)

            VStack(alignment: .leading, spacing: 4) {
                Text("Say any command while dictating or right after you finish.")
                Text("Commands automatically target your selection when you have one.")
            }
            .font(.system(size: 10, design: .rounded))
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(width: 320)
    }

    private func hudCommandGroup(title: String, commands: [String]) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title.uppercased())
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .tracking(0.8)
                .foregroundStyle(.secondary)
            HudCommandFlow(commands: commands, accent: JunoDesignTokens.accent)
        }
    }

    // MARK: - Footer

    private var showFooter: Bool {
        switch bodyKind {
        case .done, .action, .actionWorking: return false
        default:    return true
        }
    }

    @ViewBuilder
    private var footer: some View {
        HStack(spacing: 10) {
            micLabel
            if let mode = controller.currentModeLabel, !mode.isEmpty,
               case .listening = bodyKind {
                modeChip(mode)
            }
            if controller.editingSelectionCharCount > 0,
               case .listening = bodyKind {
                selectionChip(charCount: controller.editingSelectionCharCount)
            }
            Spacer(minLength: 8)
            footerHints
        }
    }

    private func selectionChip(charCount: Int) -> some View {
        HStack(spacing: 4) {
            Image(systemName: "text.cursor")
                .font(.system(size: 9, weight: .semibold))
            Text("Editing \(charCount) char\(charCount == 1 ? "" : "s")")
                .font(.system(size: 9.5, weight: .semibold, design: .rounded))
        }
        .foregroundStyle(JunoDesignTokens.accent)
        .padding(.horizontal, 7)
        .padding(.vertical, 2.5)
        .background(Capsule(style: .continuous).fill(JunoDesignTokens.accent.opacity(0.16)))
        .help("Juno will operate on your selection — say \u{201C}fix that\u{201D}, \u{201C}make that shorter\u{201D}, or \u{201C}translate that to French\u{201D}.")
    }

    private var micLabel: some View {
        HStack(spacing: 6) {
            if let target = controller.targetApp {
                if let icon = target.icon {
                    Image(nsImage: icon)
                        .resizable()
                        .interpolation(.high)
                        .frame(width: 14, height: 14)
                        .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
                } else {
                    Image(systemName: "app.fill")
                        .font(.system(size: 9.5))
                        .foregroundStyle(Color.white.opacity(0.50))
                }
                Text(truncatedTargetName(target.name))
                    .font(.system(size: 10.5, weight: .medium, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.55))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 120, alignment: .leading)
            } else {
                Image(systemName: "j.circle.fill")
                    .font(.system(size: 9.5))
                    .foregroundStyle(Color.white.opacity(0.40))
                Text("Juno")
                    .font(.system(size: 10.5, weight: .medium, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.45))
                    .lineLimit(1)
                    .frame(maxWidth: 120, alignment: .leading)
            }
        }
    }

    private func truncatedTargetName(_ name: String) -> String {
        guard name.count > 16 else { return name }
        let head = name.prefix(8)
        let tail = name.suffix(7)
        return "\(head)…\(tail)"
    }

    private func modeChip(_ label: String) -> some View {
        Text(label)
            .font(.system(size: 9.5, weight: .semibold, design: .rounded))
            .foregroundStyle(Color.white.opacity(0.72))
            .padding(.horizontal, 7)
            .padding(.vertical, 2.5)
            .background(Capsule(style: .continuous).fill(Color.white.opacity(0.10)))
    }

    @ViewBuilder
    private var footerHints: some View {
        switch bodyKind {
        case .listening:
            // The stop affordance lives on the stripe's Stop button (with its
            // shortcut keycap); repeating it here read as two different actions.
            JunoShortcutHint(label: "Cancel", keycap: "esc", onTap: { controller.cancelDictation() })
        case .refining:
            JunoShortcutHint(label: "Cancel", keycap: "esc", onTap: { controller.cancelDictation() })
        case .copyReady:
            HStack(spacing: 10) {
                JunoShortcutHint(label: "Copy", keycap: "⌘C")
                JunoShortcutHint(label: "Dismiss", keycap: "esc", onTap: { controller.cancelDictation() })
            }
        case .error:
            JunoShortcutHint(label: "Dismiss", keycap: "esc", onTap: { controller.cancelDictation() })
        case .done, .action, .actionWorking:
            EmptyView()
        }
    }

    private func stopHintLabels() -> (label: String, keycap: String) {
        switch JunoShortcutPreference.stored {
        case .fn:           return ("Stop", "fn")
        case .rightCommand: return ("Stop", "⌘")
        case .rightOption:  return ("Stop", "⌥")
        case .optionSpace:  return ("Stop", "⌥ Space")
        case .controlSpace: return ("Stop", "⌃ Space")
        }
    }

    // MARK: - Auxiliary views

    private var listeningTimer: some View {
        TimelineView(.periodic(from: controller.dictationStartedAt ?? .now, by: 1)) { ctx in
            let start = controller.dictationStartedAt ?? ctx.date
            let sec   = max(0, Int(ctx.date.timeIntervalSince(start)))
            Text(String(format: "%d:%02d", sec / 60, sec % 60))
                .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                .foregroundStyle(Color.white.opacity(0.32))
        }
    }

    private var refiningElapsedView: some View {
        TimelineView(.periodic(from: controller.refiningStartedAt ?? .now, by: 1)) { ctx in
            let start = controller.refiningStartedAt ?? ctx.date
            let sec = max(0, Int(ctx.date.timeIntervalSince(start)))
            Text(String(format: "%d:%02d", sec / 60, sec % 60))
                .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                .foregroundStyle(Color.white.opacity(0.32))
        }
    }

    private var errorTitle: String {
        if case .blocked = controller.hudState { return "Blocked" }
        return "Something went wrong"
    }

    private var errorMessage: String {
        switch controller.hudState {
        case .error(.micNoAudio):
            return "Juno is not receiving microphone audio. Check permission or try another input."
        case .error(.transcribeFailed):
            return "Couldn't transcribe this take. Try again."
        case .blocked(.axPermissionMissing):
            return "Re-enable Accessibility for this Juno build, then relaunch."
        case .blocked(let reason):
            // Forward-compat: an unrecognised `blocked:*` from the engine
            // still renders by humanising its raw suffix.
            return reason.rawValue
                .replacingOccurrences(of: "_", with: " ")
                .capitalized
        default:
            return "We couldn't transcribe this take. Try again."
        }
    }

    // MARK: - Island background

    /// Delegates to the shared `JunoIslandBackground` so the compact HUD renders
    /// against the exact same matte capsule shell. Keeping the private method
    /// preserves the byte-identical call site inside `mainIsland`.
    @ViewBuilder
    private func islandBackground(danger: Bool) -> some View {
        JunoIslandBackground(danger: danger)
    }

    // MARK: - Animation triggers

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
        let now = Date()
        guard now.timeIntervalSince(lastWordBeatAt) >= 0.2 else { return }
        lastWordBeatAt = now
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

    /// Magical replace pulse — a single soft glow that fades in over
    /// ~140ms then back out over ~180ms when the engine swaps preview
    /// text for the corrected/final version. Guards against retrigger
    /// during the same correction (one pulse per generation), and is
    /// fully suppressed under Reduce Motion so accessibility-sensitive
    /// users don't get a flash they didn't ask for.
    private func triggerCorrectionPulse(generation: Int) {
        guard generation != correctionPulseGeneration, generation > 0 else { return }
        correctionPulseGeneration = generation
        guard !reduceMotion else { return }
        withAnimation(.easeOut(duration: 0.14)) {
            correctionPulseActive = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.14) {
            // Snapshot the generation we started for so a rapid second
            // correction can't have its fade-out clobbered by ours.
            let startedFor = generation
            withAnimation(.easeIn(duration: 0.18)) {
                if startedFor == correctionPulseGeneration {
                    correctionPulseActive = false
                }
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

// MARK: - Shortcut keycap hint

private struct JunoShortcutHint: View {
    let label: String
    let keycap: String
    var onTap: (() -> Void)? = nil

    var body: some View {
        Group {
            if let onTap {
                Button(action: onTap) { hintContent }
                    .buttonStyle(.plain)
            } else {
                hintContent
            }
        }
    }

    private var hintContent: some View {
        HStack(spacing: 5) {
            Text(label)
                .font(.system(size: 10, weight: .medium, design: .rounded))
                .foregroundStyle(Color.white.opacity(0.50))
            JunoKeycap(label: keycap)
        }
    }
}

private struct LiveTranscriptWord: View {
    let span: HUDTranscriptSpan
    let reduceMotion: Bool

    var body: some View {
        Text(span.text)
            .font(.system(size: 13, weight: weight, design: .rounded))
            .foregroundStyle(foreground)
            .opacity(opacity)
            .offset(y: span.changed && !reduceMotion ? -1 : 0)
            .animation(reduceMotion ? nil : .easeOut(duration: 0.16), value: span.changed)
    }

    /// LocalAgreement-2 visual contract:
    /// - `.committed`: confirmed by two consecutive Whisper passes. Full
    ///   opacity, regular weight. NEVER shrinks (HypothesisBuffer invariant).
    /// - `.tail`: legacy/debug volatile hypothesis styling. The production
    ///   engine preview path keeps ASR tail out of the HUD entirely.
    /// - `.corrected`: post-final Qwen-adjudicated text. Full opacity with a
    ///   subtle weight bump on the changed words. Triggered only on stop.
    /// - `.draft` / `.pending`: legacy pre-engine preview / pending states.
    private var foreground: Color {
        switch span.origin {
        case .committed:
            return Color.white.opacity(0.94)
        case .tail:
            return Color.white.opacity(0.55)
        case .corrected:
            return span.changed ? Color.white : Color.white.opacity(0.94)
        case .draft:
            return Color.white.opacity(0.86)
        case .pending:
            return Color.white.opacity(0.74)
        }
    }

    private var opacity: Double {
        // Tail handles its dimming via foreground color; opacity stays at 1
        // so SwiftUI's word transitions animate smoothly. Pending state keeps
        // the legacy 0.74 dim.
        span.origin == .pending ? 0.74 : 1.0
    }

    private var weight: Font.Weight {
        switch span.origin {
        case .corrected:
            return span.changed ? .semibold : .medium
        case .committed, .draft:
            return .medium
        case .tail:
            return .regular
        case .pending:
            return .regular
        }
    }
}

private struct JunoKeycap: View {
    let label: String

    var body: some View {
        Text(label)
            .font(.system(size: 9.5, weight: .semibold, design: .rounded))
            .foregroundStyle(Color.white.opacity(0.82))
            .padding(.horizontal, 6)
            .padding(.vertical, 2.5)
            .background(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(Color.white.opacity(0.12))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.18), lineWidth: 0.5)
            )
    }
}

// `ProcessingDots` and `JunoScanShimmer` live in `JunoBrandIslandShared.swift`
// so the compact HUD can render the same primitives without duplication.

// MARK: - HUD command reference data

private let kHudInMomentCommands: [String] = [
    "scratch that", "undo that", "delete that",
    "delete last word", "delete last two words", "delete last sentence",
    "new line", "new paragraph",
    "next bullet", "next number",
    "open quote", "close quote",
]

private let kHudRecentEditCommands: [String] = [
    "fix that", "make that shorter", "make that longer",
    "make that clearer",
    "make that more formal", "make that more casual",
    "turn that into bullets",
    "translate that to [language]",
    "replace [word] with [word]",
]

// Same surface as recent-edit commands: when you have text selected in the
// frontmost app, every recent-edit command operates on the selection
// instead of the last-committed text. This list is shown as a separate
// group in the HUD popover so users discover the selection flow.
private let kHudSelectionCommands: [String] = [
    "fix that", "make that shorter", "make that more formal",
    "translate that to [language]",
    "turn that into bullets",
    "replace [word] with [word]",
    "delete the last sentence",
]

// MARK: - HUD command chip flow

private struct HudCommandFlow: View {
    let commands: [String]
    let accent: Color

    var body: some View {
        JunoFlowLayout(spacing: 5, runSpacing: 5) {
            ForEach(commands, id: \.self) { cmd in
                Text(hudChipLabel(cmd))
                    .font(.system(size: 10.5, weight: .medium, design: .rounded))
                    .foregroundStyle(accent)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(accent.opacity(0.09)))
            }
        }
    }
}

private func hudChipLabel(_ cmd: String) -> String {
    "\u{201C}" + cmd + "\u{201D}"
}
