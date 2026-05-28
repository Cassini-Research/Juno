import SwiftUI

/// Home hero — slim, no card chrome. Real Juno comma mark + LLM-rotated
/// greeting + AI sparkle marker. Status chip on the right ONLY when
/// non-idle (warming / setup / error). The greeting hydration plumbing is
/// load-bearing — keep the `applyHydrationOutcome` / `contentRevision`
/// dance intact; the only thing that changed vs. the previous design is
/// the visual VStack.
struct JunoHomeHeroCard: View {
    @ObservedObject var greeting: JunoHomeGreetingStore
    let healthOK: Bool
    let setupReady: Bool
    let setupInstallState: String
    let setupDownloading: Bool
    let canDictate: Bool
    let brokerReachable: Bool
    let homeVisitID: Int
    let isNavigating: Bool
    /// True after the first ``fetchHealth`` callback (success or failure).
    let healthProbeResolved: Bool
    let setupStatusProbeComplete: Bool
    let engineWarmingActive: Bool
    let shouldDeferSetupBadge: Bool

    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var shownHeadline: String = ""
    @State private var shownSubline: String = ""
    @State private var markDimmedForFetch: Bool = false
    @State private var lastAppliedContentRevision: Int = -1
    @State private var textOpacity: Double = 1
    @State private var sparklePulse: Bool = false
    @State private var markBreathScale: CGFloat = 1.0
    @State private var accentPulseResetTask: DispatchWorkItem?

    init(
        greeting: JunoHomeGreetingStore,
        healthOK: Bool,
        setupReady: Bool,
        setupInstallState: String,
        setupDownloading: Bool,
        canDictate: Bool,
        brokerReachable: Bool,
        homeVisitID: Int,
        isNavigating: Bool,
        healthProbeResolved: Bool,
        setupStatusProbeComplete: Bool,
        engineWarmingActive: Bool,
        shouldDeferSetupBadge: Bool
    ) {
        self.greeting = greeting
        self.healthOK = healthOK
        self.setupReady = setupReady
        self.setupInstallState = setupInstallState
        self.setupDownloading = setupDownloading
        self.canDictate = canDictate
        self.brokerReachable = brokerReachable
        self.homeVisitID = homeVisitID
        self.isNavigating = isNavigating
        self.healthProbeResolved = healthProbeResolved
        self.setupStatusProbeComplete = setupStatusProbeComplete
        self.engineWarmingActive = engineWarmingActive
        self.shouldDeferSetupBadge = shouldDeferSetupBadge

        _shownHeadline = State(initialValue: greeting.headline)
        _shownSubline = State(initialValue: greeting.subline)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            JunoCommaMark(color: JunoUI.Calm.ink)
                .frame(width: 22, height: 30)
                .opacity(markOpacity)
                .scaleEffect(markBreathScale, anchor: .center)
                .animation(JunoUI.Motion.dim, value: markOpacity)
                .animation(.easeOut(duration: 0.34), value: markBreathScale)
                .accessibilityHidden(true)
                .padding(.top, 2)

            VStack(alignment: .leading, spacing: 4) {
                Text(shownHeadline)
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)

                HStack(alignment: .firstTextBaseline, spacing: 7) {
                    Image(systemName: "sparkle")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(JunoUI.Calm.highlight)
                        .opacity(sparkleOpacity)
                        .scaleEffect(sparkleScale)
                        .animation(.easeOut(duration: 0.34), value: sparklePulse)
                        .accessibilityHidden(true)
                        .help("Greeting written fresh by Juno")
                    Text(shownSubline)
                        .font(.system(size: 12.5, weight: .regular, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .opacity(textOpacity)
            .animation(reduceMotion ? JunoUI.Motion.dim : JunoUI.Motion.cardReveal, value: shownHeadline)
            .animation(reduceMotion ? JunoUI.Motion.dim : JunoUI.Motion.cardReveal, value: shownSubline)

            Spacer(minLength: 0)
        }
        .padding(.horizontal, JunoTheme.PageInsets.detail)
        .padding(.vertical, 18)
        .frame(maxWidth: .infinity, minHeight: 78, alignment: .leading)
        // Status chip in an overlay so its appear/disappear doesn't
        // reflow the hero text — the chip floats; the row is stable.
        .overlay(alignment: .topTrailing) {
            if showTopRightStatus {
                statusPill
                    .scaleEffect(0.92, anchor: .trailing)
                    .padding(.trailing, JunoTheme.PageInsets.detail)
                    .padding(.top, 18)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .animation(JunoUI.Motion.cardReveal, value: showTopRightStatus)
        .onAppear {
            applyHydrationOutcome(greeting.hydrate(brokerReachable: brokerReachable))
            playAccentPulse()
        }
        .onChange(of: homeVisitID) { _ in
            applyHydrationOutcome(greeting.hydrate(brokerReachable: brokerReachable))
            playAccentPulse()
        }
        .onChange(of: brokerReachable) { reachable in
            applyHydrationOutcome(greeting.hydrate(brokerReachable: reachable))
        }
        .onChange(of: greeting.contentRevision) { _ in
            handleStoreContentRevisionChanged()
        }
        .onChange(of: greeting.isFetching) { fetching in
            withAnimation(.easeInOut(duration: 0.28)) {
                markDimmedForFetch = fetching
            }
        }
        .onChange(of: reduceMotion) { reduced in
            if reduced {
                shownHeadline = greeting.headline
                shownSubline = greeting.subline
                textOpacity = 1
                accentPulseResetTask?.cancel()
                accentPulseResetTask = nil
                sparklePulse = false
                markBreathScale = 1.0
            } else {
                playAccentPulse()
            }
        }
        .onDisappear {
            accentPulseResetTask?.cancel()
            accentPulseResetTask = nil
            sparklePulse = false
            markBreathScale = 1.0
            greeting.cancelDebouncedFetch()
        }
    }

    private var markOpacity: Double {
        let base = 0.95
        if greeting.isFetching && markDimmedForFetch {
            return base * 0.62
        }
        return base
    }

    /// Sparkle eases between two opacity / scale extremes when `sparklePulse`
    /// flips. With reduce-motion on, both stay at the muted endpoint so the
    /// glyph is visible but never animates.
    private var sparkleOpacity: Double {
        sparklePulse ? 1.0 : 0.55
    }
    private var sparkleScale: CGFloat {
        sparklePulse ? 1.06 : 0.94
    }

    private func playAccentPulse() {
        accentPulseResetTask?.cancel()
        accentPulseResetTask = nil
        guard !reduceMotion else {
            sparklePulse = false
            markBreathScale = 1.0
            return
        }
        withAnimation(.easeOut(duration: 0.24)) {
            sparklePulse = true
            markBreathScale = 1.035
        }
        let task = DispatchWorkItem {
            withAnimation(.easeOut(duration: 0.42)) {
                sparklePulse = false
                markBreathScale = 1.0
            }
        }
        accentPulseResetTask = task
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.72, execute: task)
    }

    private var showTopRightStatus: Bool {
        if healthOK && setupReady && canDictate && !engineWarmingActive { return false }
        return true
    }

    @ViewBuilder
    private var statusPill: some View {
        if healthOK && setupReady && canDictate {
            JunoStatusBadge(state: .ok, label: "Ready")
        } else if setupDownloading {
            JunoStatusBadge(state: .warning, label: "Downloading models…")
        } else if !canDictate {
            JunoStatusBadge(state: .warning, label: "Permissions needed")
        } else if !healthProbeResolved || !setupStatusProbeComplete || engineWarmingActive || shouldDeferSetupBadge {
            let label = engineWarmingActive ? "Voice engine starting…" : "Connecting…"
            JunoStatusBadge(state: .neutral, label: label)
        } else if !healthOK {
            JunoStatusBadge(state: .neutral, label: "Voice engine offline")
        } else if setupInstallState == "not_started" || setupInstallState == "needs_setup" {
            JunoStatusBadge(state: .warning, label: "Setup required")
        } else {
            JunoStatusBadge(state: .neutral, label: "Checking…")
        }
    }

    private func applyHydrationOutcome(_ outcome: JunoHomeGreetingStore.HydrationOutcome) {
        lastAppliedContentRevision = greeting.contentRevision
        switch outcome {
        case .snapToStore, .brokerFetchScheduled, .initialLocalTypewriter:
            snapFromStore()
        }
    }

    private func snapFromStore() {
        shownHeadline = greeting.headline
        shownSubline = greeting.subline
        textOpacity = 1
        lastAppliedContentRevision = greeting.contentRevision
    }

    private func handleStoreContentRevisionChanged() {
        guard greeting.contentRevision != lastAppliedContentRevision else { return }
        applySubtleUpdateFromStore()
    }

    private func applySubtleUpdateFromStore() {
        if isNavigating || reduceMotion {
            shownHeadline = greeting.headline
            shownSubline = greeting.subline
            textOpacity = 1
        } else {
            withAnimation(.easeOut(duration: 0.10)) {
                textOpacity = 0
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.10) {
                shownHeadline = greeting.headline
                shownSubline = greeting.subline
                withAnimation(.easeOut(duration: 0.18)) {
                    textOpacity = 1
                }
            }
        }
        lastAppliedContentRevision = greeting.contentRevision
    }
}
