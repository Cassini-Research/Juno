import AppKit
import AVFoundation
import SwiftUI

// MARK: - Sidebar items

enum MainSidebar: String, CaseIterable, Identifiable {
    case home, history, actions, voiceCommands, modes, personalization, surfacePresets, privacy, settings
    var id: String { rawValue }

    var title: String {
        switch self {
        case .home:            return "Home"
        case .actions:         return "Actions"
        case .voiceCommands:   return "Voice Commands"
        case .history:         return "History"
        case .modes:           return "Styles"
        case .personalization: return "Dictionary & Memory"
        case .surfacePresets:  return "Per-app writing"
        case .privacy:         return "Privacy"
        case .settings:        return "Settings"
        }
    }
    var symbol: String {
        switch self {
        case .home:            return "house"
        case .actions:         return "bolt.badge.checkmark"
        case .voiceCommands:   return "waveform.and.mic"
        case .history:         return "clock"
        case .modes:           return "sparkles"
        case .personalization: return "person.text.rectangle"
        case .surfacePresets:  return "app.badge.checkmark"
        case .privacy:         return "lock.shield"
        case .settings:        return "gearshape"
        }
    }
}

// MARK: - Home view

private struct JunoHomeView: View {
    @ObservedObject var surface: SurfaceEditingModel
    @ObservedObject var controller: DictationController
    @ObservedObject var greeting: JunoHomeGreetingStore
    @ObservedObject var setup: JunoSetupModel
    let homeVisitToken: Int
    let isNavigating: Bool
    /// Incremented by the shell on foreground so Home refetches ``healthz``.
    let healthRefreshTick: Int
    @ObservedObject private var windowNav = JunoMainWindowNavigator.shared
    @StateObject private var stats = JunoStatsModel()
    @ObservedObject private var perms = JunoPermissionMonitor.shared
    @ObservedObject private var lifecycle = JunoEngineLifecycle.shared
    @State private var health: BrokerHealthResponse?
    @State private var healthProbeResolved: Bool = false
    @State private var recent: [UtteranceHistoryEntry] = []
    @Environment(\.colorScheme) private var scheme

    /// Home shows the heavy "Finish setup" gate only when there's an
    /// actionable user-facing problem the lifecycle has classified — missing
    /// TCC permissions, missing models, or a hard launch failure. Pre-ready
    /// phases (.spawning/.socketBound/.healthOk) and the transient .degraded
    /// state stay on the ready grid with a thin "connecting" banner so
    /// users don't see the gate flap during cold launch.
    private var homeGateActive: Bool {
        if !perms.canDictate { return true }
        switch lifecycle.phase {
        case .needsModels, .needsPermissions, .failed:
            return true
        case .idle, .spawning, .socketBound, .healthOk,
             .modelsLoaded, .ready, .degraded:
            return false
        }
    }

    /// True only when the gate is on *and* the user actually needs to take a
    /// model-install or repair action (vs. permissions-only gate).
    private var needsFinishSetupActions: Bool {
        switch lifecycle.phase {
        case .needsModels, .failed: return true
        default: return false
        }
    }

    /// "Voice engine starting…" banner under the hero card while we are
    /// pre-ready but past spawn-attempt-start. Skipped for terminal-success
    /// phases (no banner needed) and for failure (splash owns that UI).
    private var showConnectingBanner: Bool {
        switch lifecycle.phase {
        case .idle, .spawning, .socketBound, .healthOk: return true
        default: return false
        }
    }

    private var gateTransitionAnimation: Animation {
        if JunoUserDefaults.onboardingCompleted {
            return .spring(response: 0.42, dampingFraction: 0.86)
        }
        return .easeOut(duration: 0.06)
    }

    var body: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 0) {
                JunoHomeHeroCard(
                    greeting: greeting,
                    healthOK: health?.ok == true,
                    setupReady: setup.overallReady,
                    setupInstallState: setup.installState,
                    setupDownloading: setup.installState == "downloading",
                    canDictate: perms.canDictate,
                    brokerReachable: lifecycle.phase.isReachable,
                    homeVisitID: homeVisitToken,
                    isNavigating: isNavigating,
                    healthProbeResolved: healthProbeResolved,
                    setupStatusProbeComplete: setup.hasCompletedSetupFetch,
                    engineWarmingActive: setup.engineWarmingState == "warming",
                    shouldDeferSetupBadge: setup.shouldDeferFinishSetupGate
                )
                JunoHairlineRule(.faint)
                ZStack(alignment: .topLeading) {
                    if homeGateActive {
                        JunoSetupGateView(setup: setup, surface: surface)
                            .padding(.horizontal, JunoTheme.PageInsets.detail)
                            .padding(.vertical, 14)
                    } else {
                        VStack(alignment: .leading, spacing: 0) {
                            if perms.canDictate && showConnectingBanner {
                                connectingVoiceEngineBanner
                                JunoHairlineRule(.faint)
                            }
                            // Cue card — promoted above stats: actionable comes
                            // before retrospective. The card owns its page
                            // padding *and* its trailing hairline so when no
                            // cue is visible the page closes up cleanly with
                            // no visible double-divider seam.
                            JunoActionsHomeCard {
                                controller.toggleDictation()
                            }
                            JunoScreenContextHomeCard(stats: stats)
                            JunoHomeStatsGraph(stats: stats)
                            JunoHairlineRule(.faint)
                            JunoHomeRecentList(
                                entries: Array(recent.prefix(4)),
                                onOpen: { id in windowNav.openHistory(utteranceId: id) },
                                onSeeAll: { windowNav.section = .history }
                            )
                        }
                    }
                }
                // Only animate the setup gate swap when the user is already on Home.
                // During sidebar navigation the entire page is transitioning; internal transitions feel janky.
                .animation(isNavigating ? nil : gateTransitionAnimation, value: homeGateActive)
            }
        }
        .transaction { tx in
            if isNavigating {
                tx.animation = nil
            }
        }
        .onAppear {
            stats.refresh()
            loadRecent()
            refetchHealth()
            // Right after onboarding the broker can still be warming when
            // home first paints — stats / recent / health all 404 and the
            // page sits empty until the user navigates and comes back.
            // Re-pull once the broker has had a beat to wake up so the
            // first impression is the populated home, not the empty one.
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.6) {
                stats.refresh()
                loadRecent()
                refetchHealth()
            }
        }
        .onChange(of: healthRefreshTick) { _ in
            refetchHealth()
        }
        .onChange(of: lifecycle.phase.isReachable) { reachable in
            if reachable {
                stats.refresh()
                loadRecent()
                refetchHealth()
            } else {
                health = nil
                healthProbeResolved = false
            }
        }
        // Refresh "Recent dictations" + stats whenever the user comes
        // back to the app. Without this, the home card shows stale data
        // from when the window was last opened — even though History
        // (which fetches on appear) shows the new entry.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            stats.refresh()
            loadRecent()
        }
    }

    private var connectingVoiceEngineBanner: some View {
        let title: String = {
            if !setup.hasCompletedSetupFetch { return "Checking voice engine…" }
            if setup.engineWarmingState == "warming" { return "Voice engine starting…" }
            return "Connecting to voice engine…"
        }()
        return HStack(spacing: 10) {
            ProgressView().controlSize(.small)
            Text(title)
                .font(.system(size: 11.5, design: .rounded))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
            Spacer(minLength: 0)
        }
        .padding(.horizontal, JunoTheme.PageInsets.detail)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func refetchHealth() {
        healthProbeResolved = false
        JunoBroker.fetchHealth { result in
            switch result {
            case .success(let h):
                health = h
            case .failure:
                health = nil
            }
            healthProbeResolved = true
        }
    }

    // The four-cell stats / recent-card / action-rail sections that used
    // to live here have moved to dedicated views:
    //   - JunoHomeStatsGraph.swift   (one consolidated graph)
    //   - JunoHomeRecentList.swift   (matches HistoryRowView pattern)
    //   - JunoActionsHomeCard.swift  (re-skinned cue card; same 4-state machine)
    //
    // The old action rail (`JunoHomeActionRail`) was used only for the
    // empty-recent fallback; the new design relies on the cue card to
    // surface "what to try next" instead, so the rail and its handler
    // helpers are gone.

    private func loadRecent() {
        // Fetch up to 4 — Home shows up to 4, History shows the full list.
        // Apple's pattern (Music Recently Played, Files Recents) sits at
        // 4–5 items per section; 3 felt deliberately scarce.
        JunoBroker.fetchHistory(limit: 4) { result in
            switch result {
            case .success(let resp):
                recent = resp.entries ?? []
            case .failure:
                recent = []
            }
        }
    }
}

// MARK: - Setup gate (non-technical onboarding in home)

private struct JunoSetupGateView: View {
    @ObservedObject var setup: JunoSetupModel
    @ObservedObject var surface: SurfaceEditingModel
    @ObservedObject private var perms = JunoPermissionMonitor.shared
    @ObservedObject private var lifecycle = JunoEngineLifecycle.shared
    @Environment(\.colorScheme) private var scheme

    private var laneItems: [JunoSetupLaneViewModel] {
        JunoSetupPresentation.laneItems(from: setup).filter { $0.required || $0.role != .writer }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Finish setup")
                .font(.system(.title3, design: .rounded).weight(.semibold))
                .foregroundStyle(JunoTheme.primaryText(scheme))

            VStack(spacing: 8) {
                setupStep(
                    number: 1,
                    title: "Allow access",
                    detail: "Microphone + Accessibility",
                    done: perms.canDictate
                ) {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 8) {
                            if perms.micStatus != .authorized {
                                if perms.micStatus == .notDetermined {
                                    Button("Allow microphone") { perms.requestMic() }
                                        .junoPrimaryActionButton()
                                } else {
                                    Button("Open Microphone privacy") { perms.openMicSettings() }
                                        .junoPrimaryActionButton()
                                }
                            }
                            if !perms.axGranted {
                                Button("Open Accessibility") {
                                    perms.openAccessibilitySettings()
                                }
                                .buttonStyle(.bordered).controlSize(.small)
                            }
                        }
                        HStack(spacing: 8) {
                            let screenContextReady = perms.screenContextEnabled && perms.screenRecordingGranted
                            Image(systemName: screenContextReady ? "checkmark.circle.fill" : "viewfinder")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(screenContextReady ? JunoDesignTokens.meadow : JunoTheme.secondaryText(scheme))
                            Text(screenContextReady ? "Visible screen text is enabled." : "Visible screen text is optional.")
                                .font(.caption)
                                .foregroundStyle(JunoTheme.secondaryText(scheme))
                            if !screenContextReady {
                                Button("Open Screen Recording") {
                                    perms.requestScreenRecording()
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                        }
                    }
                }
                setupStep(
                    number: 2,
                    title: "Download models",
                    detail: "Local AI — stays on your Mac",
                    done: setup.overallReady
                ) {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(laneItems) { item in
                            laneStatusRow(item)
                        }
                        if setup.bootstrapFailed {
                            bootstrapFailureCard
                        } else if setup.installState == "broker_unreachable"
                                    && laneItems.allSatisfy({ $0.ready }) {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text("Voice engine starting…")
                                    .font(.caption)
                                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                            }
                        } else if setup.installState == "downloading" {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text("Downloading models…")
                                    .font(.caption)
                                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                            }
                        } else if setup.canInstall {
                            Button("Install") { setup.triggerInstall() }
                                .junoPrimaryActionButton()
                        }
                    }
                }
                setupStep(
                    number: 3,
                    title: "Start speaking",
                    detail: "Tap shortcut · speak · tap to stop",
                    done: perms.canDictate && setup.overallReady
                ) { EmptyView() }
            }

            if !lifecycle.phase.isReachable {
                workbenchBanner
            }
        }
    }

    private func setupStep(number: Int, title: String, detail: String, done: Bool,
                           @ViewBuilder action: () -> some View) -> some View {
        HStack(alignment: .top, spacing: 14) {
            ZStack {
                Circle()
                    .fill(done ? JunoDesignTokens.meadow.opacity(0.18) : JunoDesignTokens.accent.opacity(0.12))
                    .frame(width: 32, height: 32)
                if done {
                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(JunoDesignTokens.meadow)
                } else {
                    Text("\(number)")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundStyle(JunoDesignTokens.accent)
                }
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(.headline, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(detail)
                    .font(.callout).foregroundStyle(JunoTheme.secondaryText(scheme))
                action().padding(.top, 2)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .premiumCard()
    }

    private var workbenchBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 3) {
                Text("Voice engine not connected")
                    .font(.system(.subheadline, design: .rounded).weight(.semibold))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("Juno needs the background engine running on this Mac.")
                    .font(.caption).foregroundStyle(JunoTheme.secondaryText(scheme))
            }
            Spacer()
            Button("Connection help") { JunoBrokerHelpWindow.show() }
                .junoPrimaryActionButton()
        }
        .padding(14)
        .premiumCard()
    }

    private var bootstrapFailureCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(JunoDesignTokens.danger)
                Text("Voice engine failed to start")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
            }
            if let reason = setup.bootstrapFailureReason, !reason.isEmpty {
                Text(reason)
                    .font(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 8) {
                Button("Restart engine") {
                    JunoLocalBrokerBootstrap.ensureRunningIfPossible()
                    setup.clearBootstrapFailure()
                }
                .junoPrimaryActionButton()
                Button("Get help") { JunoBrokerHelpWindow.show() }
                    .buttonStyle(.bordered).controlSize(.small)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(JunoDesignTokens.danger.opacity(0.08))
        )
    }

    private func laneStatusRow(_ item: JunoSetupLaneViewModel) -> some View {
        let label = laneStatusLabel(item)
        return HStack(alignment: .center, spacing: 8) {
            Image(systemName: item.ready ? "checkmark.circle.fill" : "arrow.down.circle")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(item.ready ? JunoDesignTokens.meadow : JunoTheme.secondaryText(scheme))
            VStack(alignment: .leading, spacing: 1) {
                Text(item.title)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(item.modelName)
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            Text(label.text)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(label.danger ? JunoDesignTokens.danger : JunoTheme.secondaryText(scheme))
        }
    }

    private func laneStatusLabel(_ item: JunoSetupLaneViewModel) -> (text: String, danger: Bool) {
        if item.ready && !setup.enginePresenceUnknown {
            return ("Ready", false)
        }
        if setup.engineWarmingState == "warming" {
            return ("Starting…", false)
        }
        if !setup.receivedSuccessfulSetupPayload && setup.enginePresenceUnknown {
            return ("Checking…", false)
        }
        if setup.enginePresenceUnknown {
            return item.ready ? ("Present (engine offline)", false) : ("Not downloaded", true)
        }
        if setup.installState == "downloading" {
            return ("Downloading…", false)
        }
        return ("Missing", true)
    }
}

// MARK: - History

private enum JunoHistoryTimeRange: Equatable, Hashable {
    case any
    case today
    case sevenDays
    case thirtyDays
    case custom(from: Date, to: Date)

    static let presets: [JunoHistoryTimeRange] = [.any, .today, .sevenDays, .thirtyDays]

    var label: String {
        switch self {
        case .any: return "All time"
        case .today: return "Today"
        case .sevenDays: return "Last 7 days"
        case .thirtyDays: return "Last 30 days"
        case .custom(let from, let to):
            let fmt = DateFormatter()
            fmt.dateFormat = "MMM d"
            return "\(fmt.string(from: from)) – \(fmt.string(from: to))"
        }
    }

    var isCustom: Bool {
        if case .custom = self { return true }
        return false
    }

    /// Lower bound — entries older than this are filtered out. Nil = no lower bound.
    var cutoff: Date? {
        let cal = Calendar.current
        let now = Date()
        switch self {
        case .any: return nil
        case .today: return cal.startOfDay(for: now)
        case .sevenDays: return cal.date(byAdding: .day, value: -7, to: now)
        case .thirtyDays: return cal.date(byAdding: .day, value: -30, to: now)
        case .custom(let from, _): return cal.startOfDay(for: from)
        }
    }

    /// Upper bound (custom range only) — entries newer than end-of-day are excluded.
    var upperBound: Date? {
        guard case .custom(_, let to) = self else { return nil }
        let cal = Calendar.current
        return cal.date(bySettingHour: 23, minute: 59, second: 59, of: to) ?? to
    }
}

/// User-configurable filter set for the History list. All four filters
/// AND together; an entry must pass every active one to appear.
private struct JunoHistoryFilters: Equatable {
    var apps: Set<String> = []      // bundleId — empty means any app
    var actions: Set<JunoActionKind> = []  // empty means any
    var time: JunoHistoryTimeRange = .any

    var isAnyActive: Bool {
        !apps.isEmpty || !actions.isEmpty || time != .any
    }
}

/// Themed dropdown filter bar. Pills are buttons that open a custom
/// popover (matched to Juno's surface tokens — not the system Menu
/// chrome). Lays out via `JunoFlowLayout` so multiple active pills wrap
/// onto a second row in the narrow split-pane instead of overflowing
/// into the detail column.
private struct JunoHistoryFilterBar: View {
    @Binding var filters: JunoHistoryFilters
    let availableApps: [(bundleId: String, displayName: String)]
    @Environment(\.colorScheme) private var scheme

    @State private var showAppPopover = false
    @State private var showActionPopover = false
    @State private var showTimePopover = false

    var body: some View {
        JunoFlowLayout(spacing: 6, runSpacing: 6) {
            appPill
            actionPill
            timePill
            if filters.isAnyActive {
                clearButton
            }
        }
    }

    // MARK: Pills

    private var appPill: some View {
        Button { showAppPopover = true } label: {
            filterPill(
                icon: "app",
                label: appPillLabel,
                isActive: !filters.apps.isEmpty
            )
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showAppPopover, arrowEdge: .bottom) {
            JunoAppFilterPopover(
                apps: availableApps,
                selected: $filters.apps
            )
        }
    }

    private var actionPill: some View {
        Button { showActionPopover = true } label: {
            filterPill(
                icon: "bolt",
                label: actionPillLabel,
                isActive: !filters.actions.isEmpty
            )
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showActionPopover, arrowEdge: .bottom) {
            JunoActionFilterPopover(selected: $filters.actions)
        }
    }

    private var timePill: some View {
        Button { showTimePopover = true } label: {
            filterPill(
                icon: "calendar",
                label: filters.time.label,
                isActive: filters.time != .any
            )
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showTimePopover, arrowEdge: .bottom) {
            JunoTimeFilterPopover(selection: $filters.time, isPresented: $showTimePopover)
        }
    }

    private var clearButton: some View {
        Button {
            filters = JunoHistoryFilters()
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 10, weight: .semibold))
                Text("Clear")
                    .font(.system(size: 11, weight: .medium, design: .rounded))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(JunoTheme.secondaryText(scheme))
        .help("Clear filters")
    }

    // MARK: Labels

    private var appPillLabel: String {
        if filters.apps.isEmpty { return "Any app" }
        if filters.apps.count == 1 {
            let key = filters.apps.first ?? ""
            return availableApps.first(where: { $0.bundleId == key })?.displayName ?? "1 app"
        }
        return "\(filters.apps.count) apps"
    }

    private var actionPillLabel: String {
        if filters.actions.isEmpty { return "Any action" }
        if filters.actions.count == 1 {
            return filters.actions.first?.descriptor.pluralName ?? "1 action"
        }
        return "\(filters.actions.count) actions"
    }

    private func filterPill(icon: String, label: String, isActive: Bool = false) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(isActive ? JunoDesignTokens.accent : JunoTheme.secondaryText(scheme))
            Text(label)
                .font(.system(size: 11.5, weight: .medium, design: .rounded))
                .lineLimit(1)
                .foregroundStyle(JunoTheme.primaryText(scheme))
            Image(systemName: "chevron.down")
                .font(.system(size: 8, weight: .bold))
                .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.7))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(isActive
                      ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.16 : 0.08)
                      : JunoTheme.elevatedCard(scheme))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(
                    isActive
                        ? JunoDesignTokens.accent.opacity(0.32)
                        : JunoTheme.border(scheme).opacity(scheme == .dark ? 0.55 : 0.16),
                    lineWidth: 0.7
                )
        )
        .contentShape(Rectangle())
    }
}

// MARK: - Filter popover building blocks

/// One row in a Juno-themed dropdown popover. Replaces the system
/// `Menu` row to keep the rest of the app's design language.
private struct JunoFilterRow: View {
    let label: String
    let isOn: Bool
    var trailing: String? = nil
    let action: () -> Void
    @Environment(\.colorScheme) private var scheme
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: isOn ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(isOn ? JunoDesignTokens.accent : JunoTheme.secondaryText(scheme).opacity(0.55))
                Text(label)
                    .font(.system(size: 12.5, weight: .medium, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .lineLimit(1)
                Spacer(minLength: 6)
                if let trailing {
                    Text(trailing)
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(hovered ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.16 : 0.08) : Color.clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 }
    }
}

private struct JunoFilterPopoverShell<Content: View>: View {
    let content: Content
    @Environment(\.colorScheme) private var scheme
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            content
        }
        .padding(6)
        .background(JunoTheme.elevatedCard(scheme))
    }
}

private struct JunoAppFilterPopover: View {
    let apps: [(bundleId: String, displayName: String)]
    @Binding var selected: Set<String>
    @State private var search = ""
    @Environment(\.colorScheme) private var scheme

    private var filtered: [(bundleId: String, displayName: String)] {
        let q = search.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !q.isEmpty else { return apps }
        return apps.filter { $0.displayName.lowercased().contains(q) }
    }

    var body: some View {
        JunoFilterPopoverShell {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                TextField("Search apps", text: $search)
                    .textFieldStyle(.plain)
                    .focusEffectDisabled()
                    .font(.system(size: 12, design: .rounded))
                if !search.isEmpty {
                    Button { search = "" } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 11))
                            .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.7))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(JunoTheme.border(scheme).opacity(scheme == .dark ? 0.18 : 0.06))
            )

            JunoFilterRow(label: "Any app", isOn: selected.isEmpty) {
                selected.removeAll()
            }
            Divider().padding(.vertical, 2)

            ScrollView(.vertical, showsIndicators: true) {
                VStack(spacing: 1) {
                    if filtered.isEmpty {
                        Text("No matching apps")
                            .font(.system(size: 11, design: .rounded))
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .padding(.vertical, 12)
                            .frame(maxWidth: .infinity)
                    } else {
                        ForEach(filtered, id: \.bundleId) { app in
                            let key = app.bundleId.isEmpty ? "__none__" : app.bundleId
                            JunoFilterRow(label: app.displayName, isOn: selected.contains(key)) {
                                if selected.contains(key) { selected.remove(key) }
                                else { selected.insert(key) }
                            }
                        }
                    }
                }
            }
            .frame(maxHeight: 240)
        }
        .frame(width: 240)
    }
}

private struct JunoActionFilterPopover: View {
    @Binding var selected: Set<JunoActionKind>
    var body: some View {
        JunoFilterPopoverShell {
            JunoFilterRow(label: "Any action", isOn: selected.isEmpty) {
                selected.removeAll()
            }
            Divider().padding(.vertical, 2)
            ForEach(JunoActionKind.allCases, id: \.self) { kind in
                JunoFilterRow(label: kind.descriptor.pluralName, isOn: selected.contains(kind)) {
                    if selected.contains(kind) { selected.remove(kind) }
                    else { selected.insert(kind) }
                }
            }
        }
        .frame(width: 200)
    }
}

private struct JunoTimeFilterPopover: View {
    @Binding var selection: JunoHistoryTimeRange
    @Binding var isPresented: Bool
    @State private var customMode: Bool = false
    @State private var fromDate: Date = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
    @State private var toDate: Date = Date()
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        JunoFilterPopoverShell {
            if !customMode {
                ForEach(Array(JunoHistoryTimeRange.presets.enumerated()), id: \.offset) { _, preset in
                    JunoFilterRow(label: preset.label, isOn: selection == preset) {
                        selection = preset
                        isPresented = false
                    }
                }
                Divider().padding(.vertical, 2)
                JunoFilterRow(
                    label: "Custom range…",
                    isOn: selection.isCustom,
                    trailing: selection.isCustom ? selection.label : nil
                ) {
                    if case .custom(let f, let t) = selection {
                        fromDate = f
                        toDate = t
                    }
                    customMode = true
                }
            } else {
                customRangeView
            }
        }
        .frame(width: customMode ? 260 : 200)
    }

    private var customRangeView: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button {
                    customMode = false
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 10, weight: .semibold))
                        Text("Back")
                            .font(.system(size: 11, weight: .medium, design: .rounded))
                    }
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
                .buttonStyle(.plain)
                Spacer()
                Text("Custom range")
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("From")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .tracking(0.8)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                DatePicker("", selection: $fromDate, in: ...toDate, displayedComponents: .date)
                    .labelsHidden()
                    .datePickerStyle(.compact)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("To")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .tracking(0.8)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                DatePicker("", selection: $toDate, in: fromDate...Date(), displayedComponents: .date)
                    .labelsHidden()
                    .datePickerStyle(.compact)
            }

            HStack(spacing: 8) {
                Button("Cancel") {
                    customMode = false
                }
                .buttonStyle(.plain)
                .font(.system(size: 11.5, weight: .medium, design: .rounded))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                Spacer()
                Button("Apply") {
                    selection = .custom(from: fromDate, to: toDate)
                    isPresented = false
                }
                .buttonStyle(.plain)
                .font(.system(size: 11.5, weight: .semibold, design: .rounded))
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 5)
                .background(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(JunoDesignTokens.accent)
                )
            }
            .padding(.top, 2)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
    }
}

private struct JunoHistoryBannerState: Equatable {
    let kind: JunoInlineStatusBanner<EmptyView>.Kind
    let title: String
    let message: String?
}

private struct JunoHistoryReprocessRequest: Equatable {
    let id = UUID()
    let utteranceId: String
}

private struct JunoHistorySplitView: View {
    @AppStorage("JunoHistoryDeleteConfirmationSuppressed") private var suppressDeleteConfirmation = false
    @ObservedObject private var windowNav = JunoMainWindowNavigator.shared
    @Environment(\.scenePhase) private var scenePhase
    @State private var entries: [UtteranceHistoryEntry] = []
    @State private var selectedId: String?
    @State private var searchText = ""
    @State private var loadError: String?
    /// Live supervisor state — flips immediately when the engine
    /// reconnects/disconnects. Used to drive the debounced
    /// ``engineOfflineDisplayed`` flag below.
    @State private var engineOffline: Bool = false
    /// What the UI actually shows. Trails ``engineOffline`` by ~1.5s so
    /// a transient supervisor flap doesn't blank the History list to a
    /// "Voice engine offline" splash and back. The debounce is short
    /// enough that real outages still surface before the user gives up.
    @State private var engineOfflineDisplayed: Bool = false
    @State private var engineOfflineDebounceTask: Task<Void, Never>?
    @State private var filters: JunoHistoryFilters = JunoHistoryFilters()
    @State private var banner: JunoHistoryBannerState?
    @State private var bannerDismissTask: Task<Void, Never>?
    @State private var deletingId: String?
    @State private var savingPhrase = false
    @State private var pendingDeleteEntry: UtteranceHistoryEntry?
    @State private var rowAudioPlayer: AVAudioPlayer?
    @State private var rowReplayingId: String?
    @State private var rowReplayLoadingId: String?
    @State private var rowReplayClearTask: Task<Void, Never>?
    @State private var reprocessRequest: JunoHistoryReprocessRequest?
    /// Cursor for the next "load older" page. ``nil`` either means we
    /// haven't loaded the first page yet or the broker said there are no
    /// older rows. Set from ``BrokerHistoryResponse.nextCursorUpdatedAtMs``
    /// or, on legacy brokers, from the last entry's pagination cursor.
    @State private var olderPageCursor: Int64?
    /// True while a paginated load is in flight; used to prevent the
    /// LazyVStack onAppear from firing duplicate fetches as the user
    /// approaches the bottom of the list.
    @State private var isLoadingOlder: Bool = false
    /// True once the broker confirms (or our fallback infers) that there
    /// are no more older pages. Stops further load-more attempts and lets
    /// the UI show an "end of history" sentinel.
    @State private var endOfHistoryReached: Bool = false
    /// Page size for both initial load and subsequent paginated loads.
    /// 80 keeps the first paint cheap; load-more then fills incrementally.
    private static let historyPageSize: Int = 80
    private let refreshTimer = Timer.publish(every: 5, on: .main, in: .common).autoconnect()
    private let engineStateNotification = NotificationCenter.default
        .publisher(for: .junoEngineSupervisorStateChanged)
    @Environment(\.colorScheme) private var scheme

    private var shouldAutoRefresh: Bool {
        scenePhase == .active || NSApp.isActive
    }

    private struct HistorySection: Identifiable {
        let id: String
        let title: String
        let entries: [UtteranceHistoryEntry]
    }

    /// Distinct app bundles seen in the loaded entries — populates the
    /// "Any app" dropdown. Sorted by display name.
    private var availableApps: [(bundleId: String, displayName: String)] {
        var seen: [String: String] = [:]
        for e in entries {
            let bid = e.context?.appBundleId ?? ""
            let name = e.context?.appName ?? ""
            let key = bid.isEmpty ? "__none__" : bid
            let display = name.isEmpty ? (bid.isEmpty ? "No app" : bid) : name
            if seen[key] == nil { seen[key] = display }
        }
        return seen.map { (bundleId: $0.key, displayName: $0.value) }
            .sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
    }

    private var filteredEntries: [UtteranceHistoryEntry] {
        var list = entries

        // App.
        if !filters.apps.isEmpty {
            list = list.filter { entry in
                let bid = entry.context?.appBundleId ?? ""
                let key = bid.isEmpty ? "__none__" : bid
                return filters.apps.contains(key)
            }
        }

        // Action kind.
        if !filters.actions.isEmpty {
            list = list.filter { entry in
                (entry.actions ?? []).contains { filters.actions.contains($0.kind) }
            }
        }

        // Time.
        if let cutoff = filters.time.cutoff {
            let cutoffMs = Int64(cutoff.timeIntervalSince1970 * 1000)
            list = list.filter { ($0.tsUnixMs ?? 0) >= cutoffMs }
        }
        if let upper = filters.time.upperBound {
            let upperMs = Int64(upper.timeIntervalSince1970 * 1000)
            list = list.filter { ($0.tsUnixMs ?? 0) <= upperMs }
        }

        // Free-text search.
        guard !searchText.isEmpty else { return list }
        let q = searchText.lowercased()
        return list.filter {
            ($0.transcript?.lowercased().contains(q) ?? false) ||
            ($0.context?.appName?.lowercased().contains(q) ?? false)
        }
    }

    private var groupedSections: [HistorySection] {
        let items = filteredEntries
        guard !items.isEmpty else { return [] }
        let cal = Calendar.current
        let now = Date()

        func bucketTitle(for tsMs: Int64?) -> String {
            guard let tsMs else { return "Earlier" }
            let d = Date(timeIntervalSince1970: TimeInterval(tsMs) / 1000.0)
            if cal.isDateInToday(d) { return "Today" }
            if cal.isDateInYesterday(d) { return "Yesterday" }
            let days = cal.dateComponents([.day], from: d, to: now).day ?? 999
            if days < 7 {
                let fmt = DateFormatter()
                fmt.locale = Locale.current
                fmt.dateFormat = "EEEE"
                return fmt.string(from: d)
            }
            return "Earlier"
        }

        var order: [String] = []
        var buckets: [String: [UtteranceHistoryEntry]] = [:]
        for e in items {
            let key = bucketTitle(for: e.tsUnixMs)
            if buckets[key] == nil { order.append(key) }
            buckets[key, default: []].append(e)
        }
        let preferred = ["Today", "Yesterday"]
        let sortedKeys = preferred.filter { buckets[$0] != nil } + order.filter { !preferred.contains($0) }
        return sortedKeys.map { k in
            HistorySection(id: k, title: k, entries: buckets[k] ?? [])
        }
    }

    /// Tail of the History list: spinner while a paginated fetch is in
    /// flight, "End of history" sentinel when the broker has no more
    /// rows. Hidden in the empty state. Lightweight — no buttons, no
    /// state of its own; the user's only paging action is to keep
    /// scrolling.
    @ViewBuilder
    private var paginationFooter: some View {
        if !entries.isEmpty {
            HStack(spacing: 8) {
                Spacer(minLength: 0)
                if isLoadingOlder {
                    ProgressView()
                        .controlSize(.small)
                    Text("Loading older…")
                        .font(.system(size: 10.5, weight: .medium, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                } else if endOfHistoryReached {
                    Text("End of history")
                        .font(.system(size: 10.5, weight: .medium, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.7))
                }
                Spacer(minLength: 0)
            }
            .padding(.vertical, 10)
        }
    }

    var body: some View {
        NavigationSplitView {
            VStack(spacing: 10) {
                JunoSplitColumnTitleRow(title: "History", trailing: {
                    Button(action: load) {
                        Image(systemName: "arrow.clockwise")
                    }
                    .junoSecondaryActionButton()
                    .help("Refresh")
                })
                JunoInlineSearchField(prompt: "Search transcripts, apps…", text: $searchText)
                JunoHistoryFilterBar(
                    filters: $filters,
                    availableApps: availableApps
                )

                Group {
                    if entries.isEmpty && loadError == nil {
                        JunoChromeEmptyState(
                            title: "No history yet",
                            message: "Your dictations on this Mac will appear here once you've used Juno.",
                            symbol: "clock"
                        )
                    } else if engineOfflineDisplayed {
                        // The engine is the source of history, capability checks
                        // and action execution. Surfacing the supervisor state here
                        // is more honest than a generic decode error and matches
                        // what's actually wrong: the local engine isn't answering.
                        // Debounced via ``engineOfflineDisplayed`` so a transient
                        // .connecting blip doesn't swap the list out for ~250ms
                        // and back — the flicker the user described.
                        JunoChromeEmptyState(
                            title: "Voice engine offline",
                            message: "Reconnecting\u{2026} history will reload automatically.",
                            symbol: "arrow.triangle.2.circlepath"
                        )
                    } else if let err = loadError {
                        JunoChromeEmptyState(
                            title: "Could not load history",
                            message: err,
                            symbol: "exclamationmark.triangle"
                        )
                    } else {
                        ScrollView(.vertical, showsIndicators: false) {
                            LazyVStack(alignment: .leading, spacing: 14) {
                                ForEach(groupedSections) { section in
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text(section.title.uppercased())
                                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                            .tracking(1.0)
                                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                                            .padding(.horizontal, 2)

                                        VStack(spacing: 0) {
                                            ForEach(Array(section.entries.enumerated()), id: \.element.id) { index, row in
                                                HistoryRowView(
                                                    entry: row,
                                                    isSelected: selectedId == row.utteranceId,
                                                    isDeleting: deletingId == row.utteranceId,
                                                    isReplaying: rowReplayingId == row.utteranceId,
                                                    isReplayLoading: rowReplayLoadingId == row.utteranceId,
                                                    onReplay: {
                                                        replayFromRow(row)
                                                    },
                                                    onReRun: {
                                                        requestReprocess(row)
                                                    }
                                                ) {
                                                    selectedId = row.utteranceId
                                                }
                                                .onAppear {
                                                    // LazyVStack only fires this when the row
                                                    // first becomes visible — perfect signal for
                                                    // "user is approaching the bottom; fetch
                                                    // the next page". Gated inside loadOlder
                                                    // so a fast scroll doesn't spam the broker.
                                                    loadOlderIfNeeded(reachedRow: row)
                                                }
                                                if index < section.entries.count - 1 {
                                                    Divider()
                                                        .padding(.leading, 48)
                                                }
                                            }
                                        }
                                        .background(
                                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                                .fill(JunoTheme.elevatedCard(scheme))
                                        )
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                                .strokeBorder(JunoTheme.border(scheme).opacity(scheme == .dark ? 0.64 : 0.14), lineWidth: 0.6)
                                        )
                                    }
                                }
                                paginationFooter
                            }
                            .padding(.bottom, 4)
                        }
                    }
                }
            }
            .junoSplitPanePadding()
            .junoSubpaneSurface()
            .onAppear {
                let offline = !JunoEngineSupervisor.shared.isOnline
                engineOffline = offline
                engineOfflineDisplayed = offline
                load()
            }
            .onReceive(refreshTimer) { _ in
                guard shouldAutoRefresh else { return }
                load()
            }
            .onReceive(engineStateNotification) { note in
                if let state = note.object as? JunoEngineSupervisor.State {
                    let online: Bool
                    if case .online = state { online = true } else { online = false }
                    let wasOffline = engineOffline
                    engineOffline = !online
                    // The moment the engine comes back, drop the stale decode
                    // error and refresh — otherwise the user sees "Could not
                    // load history" until the next 5 s timer tick.
                    if wasOffline && online {
                        loadError = nil
                        if shouldAutoRefresh {
                            load()
                        }
                    }
                    updateOfflineDisplayDebounced(offline: !online)
                }
            }
            .onChange(of: scenePhase) { newPhase in
                if newPhase == .active {
                    load()
                }
            }
            .onChange(of: filters) { _ in
                if let sid = selectedId,
                   !filteredEntries.contains(where: { $0.utteranceId == sid }) {
                    selectedId = filteredEntries.first?.utteranceId
                }
            }
            // Sticky-banner fix: any time the user picks a different row,
            // clear the transient confirmation from the prior action so it
            // doesn't bleed into the new row's detail pane. Without this,
            // "Deleted from History" lingers on the row that took the
            // selection after the delete.
            .onChange(of: selectedId) { _ in
                clearBanner()
            }
            .onDisappear {
                bannerDismissTask?.cancel()
                bannerDismissTask = nil
                rowReplayClearTask?.cancel()
                rowReplayClearTask = nil
                rowAudioPlayer?.stop()
                rowAudioPlayer = nil
                rowReplayingId = nil
                rowReplayLoadingId = nil
                engineOfflineDebounceTask?.cancel()
                engineOfflineDebounceTask = nil
            }
            .navigationSplitViewColumnWidth(
                min: JunoTheme.SplitColumns.primaryListMin,
                ideal: JunoTheme.SplitColumns.primaryListIdeal,
                max: JunoTheme.SplitColumns.primaryListMax
            )
        } detail: {
            if let item = filteredEntries.first(where: { $0.utteranceId == selectedId }) {
                HistoryDetailPane(
                    entry: item,
                    banner: banner,
                    isDeleting: deletingId == item.utteranceId,
                    isSavingPhrase: savingPhrase,
                    reprocessRequest: reprocessRequest,
                    onReprocessRequestHandled: {
                        reprocessRequest = nil
                    },
                    onSavePhrase: { phrase in
                        savePhrase(phrase, from: item)
                    },
                    onDelete: {
                        delete(item)
                    }
                )
                // Force a fresh detail-pane view when the user picks a
                // different row. Without this, ``@State phraseDraft``
                // (and any other transient state) persists from the
                // previous row — observed: clicking a long-transcript
                // row but the Save phrase field still showing the
                // previous row's pre-fill.
                .id(item.utteranceId)
            } else {
                JunoChromeEmptyState(
                    title: "Select a session",
                    message: "Choose a row to read the full transcript and context.",
                    symbol: "doc.text"
                )
            }
        }
        .alert("Delete this dictation?", isPresented: Binding(
            get: { pendingDeleteEntry != nil },
            set: { if !$0 { pendingDeleteEntry = nil } }
        )) {
            Button("Delete", role: .destructive) {
                if let entry = pendingDeleteEntry {
                    pendingDeleteEntry = nil
                    deleteConfirmed(entry)
                }
            }
            Button("Delete and don’t ask again", role: .destructive) {
                suppressDeleteConfirmation = true
                if let entry = pendingDeleteEntry {
                    pendingDeleteEntry = nil
                    deleteConfirmed(entry)
                }
            }
            Button("Cancel", role: .cancel) {
                pendingDeleteEntry = nil
            }
        } message: {
            Text("This removes the transcript and any local replay file from this Mac permanently. You won’t be able to replay or restore it from History.")
        }
    }

    /// Debounce the swap-to-offline-empty-state by ~1.5s. A transient
    /// supervisor flap (e.g. a single missed ping during a heavy ASR
    /// finalize) clears almost immediately — flipping straight into
    /// "Voice engine offline" and back is the flicker the user reported.
    /// "Coming back online" is applied immediately because the user
    /// wants their list back the moment it's available.
    private func updateOfflineDisplayDebounced(offline: Bool) {
        engineOfflineDebounceTask?.cancel()
        if !offline {
            engineOfflineDisplayed = false
            return
        }
        engineOfflineDebounceTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            guard !Task.isCancelled else { return }
            // Re-check the live flag — a flap that resolved before the
            // debounce expired must not flip the display offline.
            if engineOffline {
                engineOfflineDisplayed = true
            }
        }
    }

    private func load() {
        JunoBroker.fetchHistory(limit: Self.historyPageSize) { result in
            switch result {
            case .success(let resp):
                let fetched = resp.entries ?? []
                entries = fetched
                loadError = nil
                endOfHistoryReached = !(resp.hasMore ?? (fetched.count >= Self.historyPageSize))
                // Cursor preference order: explicit broker field, falling
                // back to the last row's updated_at on older brokers.
                if let cursor = resp.nextCursorUpdatedAtMs, cursor > 0 {
                    olderPageCursor = cursor
                } else {
                    olderPageCursor = fetched.last?.paginationCursorMs
                }

                var appliedPending = false
                if let raw = windowNav.pendingHistoryUtteranceId {
                    let pending = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                    windowNav.pendingHistoryUtteranceId = nil
                    if !pending.isEmpty, entries.contains(where: { $0.utteranceId == pending }) {
                        selectedId = pending
                        appliedPending = true
                    }
                }

                if !appliedPending {
                    if let current = selectedId,
                       !entries.contains(where: { $0.utteranceId == current }) {
                        selectedId = entries.first?.utteranceId
                    } else if selectedId == nil {
                        selectedId = entries.first?.utteranceId
                    }
                }
            case .failure(let err):
                loadError = err.localizedDescription
                entries = []
                olderPageCursor = nil
                endOfHistoryReached = false
            }
        }
    }

    /// Triggered from each row's ``onAppear`` (cheap because LazyVStack
    /// only constructs visible rows). When the row about to come into
    /// view is one of the last few we have loaded, kick off the next
    /// page. Idempotent: gated on ``isLoadingOlder`` and
    /// ``endOfHistoryReached`` so a fast scroll doesn't spam the broker.
    private func loadOlderIfNeeded(reachedRow row: UtteranceHistoryEntry) {
        guard !isLoadingOlder, !endOfHistoryReached, loadError == nil else { return }
        guard let cursor = olderPageCursor, cursor > 0 else { return }
        // Pre-fetch trigger: when the user is within 5 rows of the end.
        let lastIndex = entries.count - 1
        guard let rowIndex = entries.firstIndex(where: { $0.utteranceId == row.utteranceId }) else {
            return
        }
        guard rowIndex >= max(0, lastIndex - 4) else { return }
        isLoadingOlder = true
        JunoBroker.fetchHistory(
            limit: Self.historyPageSize,
            beforeUpdatedAtMs: cursor
        ) { result in
            DispatchQueue.main.async {
                isLoadingOlder = false
                switch result {
                case .success(let resp):
                    let page = resp.entries ?? []
                    if page.isEmpty {
                        endOfHistoryReached = true
                        olderPageCursor = nil
                        return
                    }
                    // Dedupe by utterance_id in case the broker overlaps
                    // pages by one row at the cursor boundary.
                    let known = Set(entries.map { $0.utteranceId })
                    let novel = page.filter { !known.contains($0.utteranceId) }
                    if novel.isEmpty {
                        endOfHistoryReached = true
                        olderPageCursor = nil
                        return
                    }
                    entries.append(contentsOf: novel)
                    if let nextCursor = resp.nextCursorUpdatedAtMs, nextCursor > 0 {
                        olderPageCursor = nextCursor
                    } else {
                        olderPageCursor = novel.last?.paginationCursorMs
                    }
                    if !(resp.hasMore ?? true) {
                        endOfHistoryReached = true
                    }
                case .failure:
                    // Soft fail: leave the cursor in place so a later
                    // scroll can retry. Don't surface a banner — the
                    // first page is still visible.
                    break
                }
            }
        }
    }

    private func savePhrase(_ phrase: String, from entry: UtteranceHistoryEntry) {
        let trimmed = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            setBanner(JunoHistoryBannerState(kind: .warning, title: "Nothing to save", message: "Select a word or phrase from the transcript first."))
            return
        }
        savingPhrase = true
        clearBanner()
        JunoBroker.postJSON(
            path: "api/broker/memory/vocab",
            payload: ["term": trimmed, "canonical_form": trimmed]
        ) { obj in
            savingPhrase = false
            let ok = (obj["ok"] as? Bool) ?? false
            let skipped = (obj["skipped"] as? Bool) ?? false
            let errorCode = (obj["error_code"] as? String) ?? (obj["error"] as? String) ?? ""
            if ok && skipped {
                setBanner(JunoHistoryBannerState(
                    kind: .warning,
                    title: "Already learned",
                    message: "“\(trimmed)” is already in Dictionary & Memory."
                ))
            } else if ok {
                setBanner(JunoHistoryBannerState(
                    kind: .success,
                    title: "Saved to Dictionary & Memory",
                    message: "“\(trimmed)” is now available for future dictations."
                ))
            } else if errorCode == "protected_term" {
                setBanner(JunoHistoryBannerState(
                    kind: .warning,
                    title: "Already known to Juno",
                    message: "Built-in Juno terms do not need to be taught again."
                ))
            } else if errorCode == "vocab_conflict" {
                setBanner(JunoHistoryBannerState(
                    kind: .warning,
                    title: "Already learned differently",
                    message: "Open Dictionary & Memory to review the existing entry before changing it."
                ))
            } else {
                setBanner(JunoHistoryBannerState(
                    kind: .danger,
                    title: "Couldn’t save phrase",
                    message: (obj["error"] as? String) ?? "Try again in a moment."
                ))
            }
        }
    }

    private func requestReprocess(_ entry: UtteranceHistoryEntry) {
        guard entry.replayAvailable == true else {
            setBanner(JunoHistoryBannerState(
                kind: .warning,
                title: "Re-run unavailable",
                message: "This history row does not have retained audio."
            ))
            return
        }
        selectedId = entry.utteranceId
        reprocessRequest = JunoHistoryReprocessRequest(utteranceId: entry.utteranceId)
    }

    private func replayFromRow(_ entry: UtteranceHistoryEntry) {
        guard entry.replayAvailable == true else {
            setBanner(JunoHistoryBannerState(
                kind: .warning,
                title: "Replay unavailable",
                message: "This history row does not have retained audio."
            ))
            return
        }

        if rowReplayLoadingId == entry.utteranceId {
            rowReplayLoadingId = nil
            return
        }

        if rowReplayingId == entry.utteranceId, rowAudioPlayer?.isPlaying == true {
            rowAudioPlayer?.stop()
            rowAudioPlayer = nil
            rowReplayingId = nil
            rowReplayClearTask?.cancel()
            rowReplayClearTask = nil
            return
        }

        rowAudioPlayer?.stop()
        rowAudioPlayer = nil
        rowReplayingId = nil
        rowReplayLoadingId = entry.utteranceId
        rowReplayClearTask?.cancel()
        rowReplayClearTask = nil
        clearBanner()

        JunoBroker.fetchBinary(path: "api/broker/audio/\(entry.utteranceId)/replay") { result in
            guard rowReplayLoadingId == entry.utteranceId else { return }
            rowReplayLoadingId = nil
            switch result {
            case .failure(let error):
                setBanner(JunoHistoryBannerState(
                    kind: .danger,
                    title: "Replay failed",
                    message: error.localizedDescription
                ))
            case .success(let data):
                do {
                    let player = try AVAudioPlayer(data: data)
                    player.prepareToPlay()
                    guard player.play() else {
                        setBanner(JunoHistoryBannerState(
                            kind: .danger,
                            title: "Replay failed",
                            message: "The retained recording could not be played."
                        ))
                        return
                    }
                    rowAudioPlayer = player
                    rowReplayingId = entry.utteranceId
                    let durationNs = UInt64(max(player.duration, 0.5) * 1_000_000_000)
                    let utteranceId = entry.utteranceId
                    rowReplayClearTask = Task { @MainActor in
                        try? await Task.sleep(nanoseconds: durationNs)
                        guard !Task.isCancelled, rowReplayingId == utteranceId else { return }
                        rowAudioPlayer = nil
                        rowReplayingId = nil
                    }
                } catch {
                    setBanner(JunoHistoryBannerState(
                        kind: .danger,
                        title: "Replay failed",
                        message: error.localizedDescription
                    ))
                }
            }
        }
    }

    /// Set a transient banner. Success/info banners auto-dismiss after a
    /// few seconds; danger/warning banners persist until the user takes
    /// another action (so they have time to read the error). Selecting
    /// another row also clears whatever banner is up.
    private func setBanner(_ b: JunoHistoryBannerState?) {
        bannerDismissTask?.cancel()
        bannerDismissTask = nil
        banner = b
        guard let b, b.kind == .success else { return }
        let task = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 4_000_000_000)
            guard !Task.isCancelled else { return }
            banner = nil
        }
        bannerDismissTask = task
    }

    private func clearBanner() {
        bannerDismissTask?.cancel()
        bannerDismissTask = nil
        banner = nil
    }

    private func delete(_ entry: UtteranceHistoryEntry) {
        if suppressDeleteConfirmation {
            deleteConfirmed(entry)
        } else {
            pendingDeleteEntry = entry
        }
    }

    private func deleteConfirmed(_ entry: UtteranceHistoryEntry) {
        deletingId = entry.utteranceId
        clearBanner()
        JunoBroker.deleteJSON(path: "api/broker/history/\(entry.utteranceId)") { obj in
            deletingId = nil
            let ok = (obj["ok"] as? Bool) ?? false
            let error = (obj["error"] as? String) ?? ""
            if ok || error == "not_found" {
                removeEntryLocally(entry.utteranceId)
                setBanner(JunoHistoryBannerState(
                    kind: .success,
                    title: error == "not_found" ? "Already removed" : "Deleted from History",
                    message: error == "not_found"
                        ? "That dictation was already gone, so the list was cleaned up."
                        : "The dictation and any local replay file were removed from this Mac."
                ))
            } else {
                setBanner(JunoHistoryBannerState(kind: .danger, title: "Delete failed", message: error.isEmpty ? "Try again in a moment." : error))
            }
        }
    }

    private func removeEntryLocally(_ utteranceId: String) {
        let currentFiltered = filteredEntries
        let removedIndex = currentFiltered.firstIndex(where: { $0.utteranceId == utteranceId })
        let remaining = entries.filter { $0.utteranceId != utteranceId }
        entries = remaining
        guard selectedId == utteranceId else { return }
        if let index = removedIndex {
            let filteredRemaining = filteredEntries.filter { $0.utteranceId != utteranceId }
            if index < filteredRemaining.count {
                selectedId = filteredRemaining[index].utteranceId
            } else {
                selectedId = filteredRemaining.last?.utteranceId
            }
        } else {
            selectedId = filteredEntries.first?.utteranceId
        }
    }
}

private struct HistoryRowView: View {
    let entry: UtteranceHistoryEntry
    let isSelected: Bool
    let isDeleting: Bool
    let isReplaying: Bool
    let isReplayLoading: Bool
    let onReplay: () -> Void
    let onReRun: () -> Void
    let onSelect: () -> Void
    @State private var isHovered = false
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Button(action: onSelect) {
                HStack(alignment: .top, spacing: 10) {
                    appIcon

                    VStack(alignment: .leading, spacing: 4) {
                        // App-name text dropped — the icon column on the
                        // left already identifies the app. The body
                        // preview is the row's headline now.
                        Text(entry.historyPrimaryLine)
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)
                            .font(.system(size: 12, weight: .semibold, design: .rounded))
                            .foregroundStyle(JunoTheme.primaryText(scheme))

                        Text(entry.historySecondaryLine)
                            .font(.system(size: 9.5, weight: .medium, design: .rounded))
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .lineLimit(1)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            VStack(alignment: .trailing, spacing: 6) {
                Text(entry.historyTimestampLabel)
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))

                if isDeleting {
                    ProgressView()
                        .controlSize(.small)
                } else if isHovered || isReplaying || isReplayLoading {
                    // Row keeps just Replay. Re-run also lives in the
                    // detail-pane header for the selected row, where it
                    // has more room — duplicating it on every row was
                    // crowding the 56pt right column on narrow sidebars
                    // and made the chips wrap. One affordance per row,
                    // both controls visible together once a row is open.
                    let replayOn = entry.replayAvailable == true
                    if isReplayLoading {
                        ProgressView()
                            .controlSize(.mini)
                            .frame(width: 20, height: 20)
                            .help("Loading replay")
                    } else {
                        rowActionButton(
                            systemName: isReplaying ? "stop.fill" : "play.fill",
                            help: replayOn
                                ? (isReplaying ? "Stop replay" : "Replay audio")
                                : "Audio wasn\u{2019}t retained for this session \u{2014} turn on audio retention in Settings \u{2192} Storage to enable replay.",
                            isEnabled: replayOn,
                            action: onReplay
                        )
                    }
                }
            }
            .frame(width: 44, alignment: .trailing)
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(isSelected ? JunoTheme.secondaryText(scheme).opacity(scheme == .dark ? 0.10 : 0.055) : Color.clear)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(
                    isSelected ? JunoTheme.border(scheme).opacity(0.65) : Color.clear,
                    lineWidth: 0.6
                )
        )
        .onHover { hovering in
            isHovered = hovering
        }
    }

    private func rowActionButton(
        systemName: String,
        help: String,
        isEnabled: Bool = true,
        action: @escaping () -> Void
    ) -> some View {
        Button {
            guard isEnabled else { return }
            action()
        } label: {
            Image(systemName: systemName)
                .font(.system(size: 9.5, weight: .semibold))
                .frame(width: 20, height: 20)
                .foregroundStyle(isEnabled
                    ? JunoDesignTokens.accent
                    : JunoTheme.secondaryText(scheme).opacity(0.5))
                .background(
                    Circle().fill(
                        isEnabled
                            ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.16 : 0.10)
                            : JunoTheme.secondaryText(scheme).opacity(scheme == .dark ? 0.10 : 0.06)
                    )
                )
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .junoNoFocusRing()
        .accessibilityLabel(Text(help))
        .help(help)
    }

    @ViewBuilder
    private var appIcon: some View {
        if entry.isActionHistoryRow {
            Image(systemName: "bolt.badge.checkmark")
                .font(.system(size: 10.5, weight: .semibold))
                .foregroundStyle(JunoDesignTokens.accent)
                .frame(width: 18, height: 18)
                .background(
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(JunoDesignTokens.accent.opacity(scheme == .dark ? 0.18 : 0.10))
                )
                .help("Voice Action")
        } else {
            let bundleId = entry.context?.appBundleId
            let url = bundleId.flatMap { NSWorkspace.shared.urlForApplication(withBundleIdentifier: $0) }
            let img = url.map { NSWorkspace.shared.icon(forFile: $0.path) }
                ?? NSImage(systemSymbolName: "app", accessibilityDescription: nil)
                ?? NSImage()
            Image(nsImage: img)
                .resizable()
                .scaledToFit()
                .frame(width: 18, height: 18)
                .cornerRadius(4)
        }
    }

}

private struct HistoryDetailPane: View {
    let entry: UtteranceHistoryEntry
    @ObservedObject private var windowNav = JunoMainWindowNavigator.shared
    @ObservedObject private var actionExecutor = JunoActionExecutor.shared
    let banner: JunoHistoryBannerState?
    let isDeleting: Bool
    let isSavingPhrase: Bool
    let reprocessRequest: JunoHistoryReprocessRequest?
    let onReprocessRequestHandled: () -> Void
    let onSavePhrase: (String) -> Void
    let onDelete: () -> Void
    @Environment(\.colorScheme) private var scheme
    @State private var audioPlayer: AVAudioPlayer?
    @State private var audioError:  String?
    @State private var showRaw:     Bool = false
    @State private var phraseDraft = ""
    @State private var showSavePhrasePopover = false
    /// LLM-extracted vocab candidates. ``nil`` while the writer is being
    /// asked; empty array means it answered "nothing qualifies"; a
    /// populated array replaces the regex heuristic.
    @State private var llmCandidates: [String]? = nil
    @State private var llmCandidatesLoading = false
    @State private var knownVocabularyTerms: Set<String> = MemoryStoreViewModel.protectedVocabTerms
    @State private var knownVocabularyLoading = false
    @State private var copiedTranscript = false
    @State private var copiedTranscriptResetTask: Task<Void, Never>?

    // For action entries the transcript and rewrite diff are demoted to
    // disclosure links. These flips track inline expansion. Reset when
    // the user switches to a different entry.
    @State private var showTranscriptInline: Bool = false
    @State private var showRewriteInline:    Bool = false

    // Provenance section ("What you said") expansion. The dictation
    // path auto-opens this when Juno rewrote anything (the user came
    // here to see the diff). Action entries leave it collapsed by
    // default — the artifact above is the hero, not the transcript.
    // Tracked separately from the legacy `show*Inline` flags so the
    // two histories don't fight when both render paths coexist.
    @State private var showProvenance: Bool = false
    @State private var provenanceDefaultedFor: String? = nil

    // Hero card "View full" toggle for long transcripts. False by
    // default ⇒ body scrolls inside the 320pt cap. True ⇒ cap is
    // removed and the body expands to its full height inside the
    // outer page scroll.
    @State private var heroExpandedFull: Bool = false

    // Per-entry Copy feedback for the in-hero primary Copy button.
    // Separate from `copiedTranscript` (overflow-menu) and
    // `copiedFromBar` (sticky action bar) so the three locations can
    // show their own "Copied" feedback without interfering.
    @State private var copiedFromHero: Bool = false
    @State private var copiedFromHeroResetTask: Task<Void, Never>?

    // MARK: Re-process state
    @State private var showReprocessSheet = false
    @State private var reprocessModeList: [(id: String, label: String, isCustom: Bool)] = []
    @State private var selectedReprocessMode = ""
    @State private var isReprocessing = false
    @State private var reprocessResult: String? = nil
    @State private var reprocessResultModeLabel = ""
    @State private var reprocessError: String? = nil
    @State private var isRetryingActions = false
    @State private var retryActionResult: String? = nil
    @State private var retryActionError: String? = nil

    // MARK: Insert-again + diagnostics state
    @State private var isInsertingAgain = false
    @State private var insertAgainError: String? = nil
    @State private var insertAgainSuccess: String? = nil
    @State private var showDiagnostics = false
    @State private var copiedFromBar = false
    @State private var copiedFromBarResetTask: Task<Void, Never>?

    private var savePhraseCandidates: [String] {
        filterLearningCandidates(llmCandidates ?? entry.vocabularyCandidates)
    }

    private var phraseDraftIssue: String? {
        let trimmed = phraseDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard isLearnableTermShape(trimmed) else {
            return "Save one word or short term, not a sentence."
        }
        guard let key = learningTermKey(trimmed) else { return "Choose a term to save." }
        if MemoryStoreViewModel.protectedVocabTerms.contains(key) {
            return "Juno already knows that term."
        }
        if knownVocabularyTerms.contains(key) {
            return "Already in Dictionary & Memory."
        }
        return nil
    }

    private var displayedActionResults: [JunoActionResult] {
        entry.actionResultsForHistoryDisplay(activeUtteranceId: actionExecutor.inFlight?.utteranceId)
    }

    var body: some View {
        // Sticky action toolbar lives OUTSIDE the ScrollView in the same
        // VStack so it stays pinned to the top of the detail column for
        // any transcript length. This deliberately matches the pattern
        // ``JunoHistorySplitView`` already uses on the left column
        // (title row + search field + filter bar above its ScrollView).
        // We previously tried ``.safeAreaInset(edge: .top)`` here — works
        // visually, but it's a different SwiftUI mechanism than the rest
        // of this app's chrome. Keeping the layout shape identical to a
        // surface that has shipped without overlay/hit-test bugs is the
        // safer choice; it also means there's no inset-hit-test surface
        // outside the bar's visible bounds.
        // No top sticky action bar any more — Copy moved into the hero
        // card next to the transcript it copies, Replay / Re-run /
        // Diagnostics demoted to quiet footer links, and the "Insert
        // again" recovery affordance lives on the recovery strip where
        // the failure narrative is. Three header chunks (sticky bar +
        // sessionHeader + hero) before any content was the actual
        // problem the redesign was meant to fix.
        VStack(spacing: 0) {
            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 16) {

                    sessionHeader
                    recoveryStrip
                    actionFeedbackBanners
                    contentSections
                    footerActions
                    Spacer(minLength: 12)
                    diagnosticsDisclosure

                }
                .junoDetailPagePadding()
                .frame(maxWidth: 640, alignment: .leading)
            }
        }
        .sheet(isPresented: $showReprocessSheet) {
            reprocessSheet
        }
        .onChange(of: entry.utteranceId) { _ in
            reprocessResult = nil
            reprocessError = nil
            isReprocessing = false
            reprocessResultModeLabel = ""
            isRetryingActions = false
            retryActionResult = nil
            retryActionError = nil
            copiedTranscript = false
            copiedTranscriptResetTask?.cancel()
            copiedTranscriptResetTask = nil
            copiedFromBar = false
            copiedFromBarResetTask?.cancel()
            copiedFromBarResetTask = nil
            isInsertingAgain = false
            insertAgainError = nil
            insertAgainSuccess = nil
            showDiagnostics = false
            showTranscriptInline = false
            showRewriteInline = false
            heroExpandedFull = false
            copiedFromHero = false
            copiedFromHeroResetTask?.cancel()
            copiedFromHeroResetTask = nil
            // Defer the provenance default to the section's onAppear /
            // onChange so per-entry-type defaults (auto-open for
            // dictation-with-rewrite, collapsed for actions) win.
            provenanceDefaultedFor = nil
        }
        .onAppear {
            handleReprocessRequestIfNeeded()
        }
        .onDisappear {
            copiedTranscriptResetTask?.cancel()
            copiedTranscriptResetTask = nil
            copiedFromBarResetTask?.cancel()
            copiedFromBarResetTask = nil
        }
        .onChange(of: reprocessRequest?.id) { _ in
            handleReprocessRequestIfNeeded()
        }
    }

    /// Main scrolling content — kept as a separate ``@ViewBuilder`` so the
    /// SwiftUI type-checker doesn't time out on the ``body`` expression
    /// (a recurring trap on a view this size). The structural rule:
    /// transcript and action artifacts ALWAYS render when present, even
    /// on failure. Failures get communicated via the recovery strip and
    /// the diagnostics disclosure, not by hiding the user's data.
    @ViewBuilder
    private var contentSections: some View {
        // ── Hero by entry type ─────────────────────────────────────
        // - Action entry: artifact cards are the hero; transcript +
        //   rewrite diff are demoted to disclosure links below.
        // - Pure dictation: transcript is the hero with rewrite diff
        //   under it.
        // The transcript ALWAYS appears when text exists, even when the
        // session failed to insert/process. We never hide user data.
        let actions = displayedActionResults
        if !actions.isEmpty {
            actionHeroSection(actions: actions)

            // Single unified "What you said" provenance section under
            // the action hero — collapsed by default since the user came
            // here to see the action that got created, not to re-read
            // their literal command. When Juno rewrote the transcript,
            // the disclosure summary shows the edit count so the user
            // knows there's something worth expanding. Replaces the
            // previous twin disclosures (Show what you said + Juno
            // changed what you said) that dragged two separately-styled
            // slabs into the page.
            if let fin = entry.transcript?.trimmingCharacters(in: .whitespacesAndNewlines),
               !fin.isEmpty {
                let rawTrim = entry.rawTranscript?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                historyDetailCard(padding: 16) {
                    provenanceSection(
                        raw: rawTrim.isEmpty ? fin : rawTrim,
                        final: fin,
                        defaultExpanded: false
                    )
                }
            }
        } else if let fin = entry.transcript?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !fin.isEmpty {
            // Single unified card: final transcript on top (the outcome),
            // raw transcript inline below as a labelled "you said" section
            // when Juno rewrote anything. Replaces the previous pair of
            // separately-styled slabs (plain card + accent-tinted card)
            // which felt visually disconnected and rendered two walls of
            // text for the same utterance.
            unifiedDictationDetailCard(
                final: fin,
                raw: entry.showsRewriteSection ? entry.rawTranscript : nil
            )
        } else if let raw = entry.rawTranscript?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !raw.isEmpty {
            // Failure path with no rewrite output but a raw STT result —
            // still surface what we heard so the user has something to
            // copy / re-insert.
            unifiedDictationDetailCard(final: raw, raw: nil)
        }
    }

    /// Inline result banners for in-flight or just-finished operations.
    /// Distinct from ``recoveryStrip``: this surface is ephemeral feedback
    /// for actions the user just took, not a description of session state.
    @ViewBuilder
    private var actionFeedbackBanners: some View {
        if let err = audioError {
            JunoInlineStatusBanner(kind: .danger, title: "Replay unavailable",
                                  message: err, systemImage: "waveform.circle")
        }
        if let banner {
            JunoInlineStatusBanner(kind: banner.kind, title: banner.title,
                                   message: banner.message,
                                   systemImage: banner.kind == .danger ? "exclamationmark.triangle" : "checkmark.circle")
        }
        if isReprocessing {
            reprocessLoadingCard
        } else if let err = reprocessError {
            JunoInlineStatusBanner(kind: .danger, title: "Re-run failed",
                                  message: err, systemImage: "arrow.triangle.2.circlepath")
        } else if let result = reprocessResult {
            reprocessResultCard(result)
        }
        if isRetryingActions {
            JunoInlineStatusBanner(kind: .info, title: "Retrying action",
                                  message: "Checking permissions and saving again…",
                                  systemImage: "arrow.clockwise")
        } else if let err = retryActionError {
            JunoInlineStatusBanner(kind: .danger, title: "Retry failed",
                                  message: err, systemImage: "exclamationmark.triangle")
        } else if let result = retryActionResult {
            JunoInlineStatusBanner(kind: .success, title: "Action retried",
                                  message: result, systemImage: "checkmark.circle")
        }
        if isInsertingAgain {
            JunoInlineStatusBanner(kind: .info, title: "Inserting…",
                                  message: "Bringing the saved transcript back to your last cursor.",
                                  systemImage: "text.cursor")
        } else if let err = insertAgainError {
            JunoInlineStatusBanner(kind: .danger, title: "Couldn’t insert",
                                  message: err, systemImage: "exclamationmark.triangle")
        } else if let msg = insertAgainSuccess {
            JunoInlineStatusBanner(kind: .success, title: "Inserted",
                                  message: msg, systemImage: "checkmark.circle")
        }
    }

    @ViewBuilder
    private func historyDetailCard<Content: View>(
        padding: CGFloat = 12,
        @ViewBuilder content: () -> Content
    ) -> some View {
        content()
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(padding)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(JunoTheme.cardBackground(scheme))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(JunoTheme.border(scheme).opacity(scheme == .dark ? 0.62 : 0.18), lineWidth: 0.6)
            )
    }

    // MARK: - Session header

    /// Quiet header: app icon, app name, timestamp, single overflow menu.
    /// Active per-entry controls (replay / re-run) live in the footer;
    /// management actions (copy, delete) live in this overflow menu — the
    /// macOS Mail / Notes / Reminders pattern. The previous design packed
    /// four colored buttons in the right of the header, which forced the
    /// app name to truncate to "Goog…" on a typical detail-pane width.
    private var sessionHeader: some View {
        HStack(alignment: .center, spacing: 14) {
            appIconView
                .frame(width: 40, height: 40)
                .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))

            VStack(alignment: .leading, spacing: 2) {
                Text(entry.historyHeaderTitle)
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .lineLimit(1)
                if let subtitle = entry.historyHeaderSubtitle {
                    Text(subtitle)
                        .font(.system(size: 12, weight: .regular, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .lineLimit(1)
                }
            }

            Spacer(minLength: 8)

            headerOverflowMenu
        }
        .padding(.vertical, 4)
    }

    private var headerOverflowMenu: some View {
        Menu {
            Button {
                copyTranscriptToPasteboard()
            } label: {
                Label(copiedTranscript ? "Copied" : "Copy transcript", systemImage: copiedTranscript ? "checkmark" : "doc.on.doc")
            }
            .disabled(transcriptForCopy == nil)

            Divider()

            Button(role: .destructive) {
                onDelete()
            } label: {
                Label(isDeleting ? "Deleting…" : "Delete from History", systemImage: "trash")
            }
            .disabled(isDeleting)
        } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 13, weight: .semibold))
                .frame(width: 32, height: 28)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .background(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(JunoTheme.secondaryText(scheme).opacity(scheme == .dark ? 0.10 : 0.06))
                )
                .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .frame(width: 36, height: 28)
        .junoNoFocusRing()
        .help("More actions")
    }

    private var transcriptForCopy: String? {
        guard let t = entry.transcript?.trimmingCharacters(in: .whitespacesAndNewlines),
              !t.isEmpty else { return nil }
        return t
    }

    // MARK: - Sticky action bar
    //
    // Pinned to the top of the detail column by living in the SAME
    // ``VStack`` as the ScrollView (above it), not as an overlay or a
    // safe-area inset. This mirrors the existing precedent in
    // ``JunoHistorySplitView``'s left column (title row + search +
    // filter bar above the list ScrollView). Recovery affordances live
    // here so they remain reachable for arbitrarily long transcripts;
    // every button is gated on whether the entry actually supports it
    // (see the broker-derived ``recovery.actions`` set), with disabled
    // buttons keeping their tooltip explaining why.
    //
    // Importantly: this introduces NO new SwiftUI machinery beyond what
    // the rest of the app already uses. No NSWindow / NSPanel, no
    // ``.overlay``, no ``.safeAreaInset``, no ``.allowsHitTesting``,
    // no full-window frames — i.e. nothing that could regress the
    // category of bug the JunoActionToastOverlay fix locked down.
    private var stickyActionBar: some View {
        let actions = recoveryActions
        let canCopy = transcriptForCopy != nil
        let canInsertAgain = actions.contains(.insertAgain)
            && !isInsertingAgain && transcriptForCopy != nil
        let replayOn = entry.replayAvailable == true
        let replayDisabledHelp = "Audio wasn’t retained for this session."
        return HStack(spacing: 6) {
            stickyBarButton(
                title: copiedFromBar ? "Copied" : "Copy",
                systemImage: copiedFromBar ? "checkmark" : "doc.on.doc",
                help: canCopy ? "Copy the saved transcript to the clipboard"
                              : "There’s no transcript to copy",
                isEnabled: canCopy
            ) {
                copyTranscriptFromBar()
            }

            if actions.contains(.insertAgain) || (entry.displayFailureReason != nil && transcriptForCopy != nil) {
                stickyBarButton(
                    title: isInsertingAgain ? "Inserting…" : "Insert again",
                    systemImage: "text.cursor",
                    help: "Put the saved transcript on the clipboard and paste it into the frontmost app.",
                    isEnabled: canInsertAgain,
                    tint: JunoDesignTokens.accent
                ) {
                    runInsertAgain()
                }
            }

            stickyBarButton(
                title: audioPlayer?.isPlaying == true ? "Stop" : "Replay",
                systemImage: audioPlayer?.isPlaying == true ? "stop.fill" : "play.fill",
                help: replayOn ? "Replay the original audio" : replayDisabledHelp,
                isEnabled: replayOn
            ) {
                if audioPlayer?.isPlaying == true { audioPlayer?.stop() }
                else { replayAudio() }
            }

            stickyBarButton(
                title: "Re-run",
                systemImage: "arrow.triangle.2.circlepath",
                help: replayOn ? "Re-run this session in a different writing style" : replayDisabledHelp,
                isEnabled: replayOn && !isReprocessing
            ) {
                loadModesForReprocess()
                showReprocessSheet = true
            }

            if !failedActionResults.isEmpty {
                stickyBarButton(
                    title: isRetryingActions
                        ? "Retrying…"
                        : (failedActionResults.count == 1 ? "Retry action" : "Retry actions"),
                    systemImage: "arrow.clockwise",
                    help: "Check permissions and retry the failed voice action.",
                    isEnabled: !isRetryingActions
                ) {
                    retryFailedActions()
                }
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            // Translucent bar that visually separates from the scrolling
            // content underneath. Uses the existing card surface so the
            // chrome stays consistent with the rest of the app.
            ZStack(alignment: .bottom) {
                JunoTheme.elevatedCard(scheme)
                Rectangle()
                    .fill(JunoTheme.border(scheme).opacity(scheme == .dark ? 0.4 : 0.18))
                    .frame(height: 0.5)
            }
        )
    }

    private func stickyBarButton(
        title: String,
        systemImage: String,
        help: String,
        isEnabled: Bool,
        tint: Color? = nil,
        action: @escaping () -> Void
    ) -> some View {
        Button {
            guard isEnabled else { return }
            action()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: systemImage)
                    .font(.system(size: 11, weight: .semibold))
                Text(title)
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .lineLimit(1)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .foregroundStyle(
                isEnabled
                    ? (tint ?? JunoTheme.primaryText(scheme))
                    : JunoTheme.secondaryText(scheme).opacity(0.45)
            )
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(
                        isEnabled
                            ? (tint?.opacity(0.10) ?? JunoTheme.secondaryText(scheme).opacity(scheme == .dark ? 0.10 : 0.06))
                            : Color.clear
                    )
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .junoNoFocusRing()
        .help(help)
    }

    // MARK: - Recovery strip
    //
    // One-line, one-button summary of what (if anything) went wrong with
    // this session. Driven by the broker-supplied ``recovery`` blob — the
    // shell never parses raw failure codes for user-facing copy. Strip is
    // hidden entirely when the session was clean and there are no failed
    // actions to surface.
    @ViewBuilder
    private var recoveryStrip: some View {
        if let summary = recoveryStripSummary {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: summary.symbol)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(summary.tint)
                    .padding(.top, 1)
                Text(summary.message)
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                if let primary = summary.primary {
                    Button(primary.title) { primary.run() }
                        .junoSecondaryActionButton()
                        .disabled(primary.isInFlight)
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(summary.tint.opacity(scheme == .dark ? 0.12 : 0.07))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(summary.tint.opacity(0.30), lineWidth: 0.6)
            )
        }
    }

    private struct RecoveryStripSummary {
        let message: String
        let symbol: String
        let tint: Color
        let primary: PrimaryButton?

        struct PrimaryButton {
            let title: String
            let isInFlight: Bool
            let run: () -> Void
        }
    }

    private var recoveryStripSummary: RecoveryStripSummary? {
        // Two independent failure sources: the entry-level failure code
        // (paste / capability / etc.) and the action-result statuses
        // (per-sink permission / sink errors). If both fire we prefer the
        // action one — it's actionable per-card below.
        let actions = recoveryActions
        if let failure = entry.displayFailureReason, !entry.isActionHistoryRow {
            let primary = primaryRecoveryButton(for: actions)
            let severity = entry.recovery?.severity ?? "warning"
            return RecoveryStripSummary(
                message: failure,
                symbol: severity == "danger" ? "exclamationmark.octagon.fill" : "exclamationmark.triangle.fill",
                tint: severity == "danger" ? JunoDesignTokens.danger : .orange,
                primary: primary
            )
        }
        if !failedActionResults.isEmpty {
            let summary = JunoActionBatchFormatter.summarize(displayedActionResults)
            return RecoveryStripSummary(
                message: summary.headline,
                symbol: "exclamationmark.triangle.fill",
                tint: .orange,
                primary: RecoveryStripSummary.PrimaryButton(
                    title: isRetryingActions ? "Retrying…" : "Retry",
                    isInFlight: isRetryingActions,
                    run: { retryFailedActions() }
                )
            )
        }
        return nil
    }

    /// Recovery actions the broker says this row supports. Falls back to
    /// a sensible default derived from local fields when the broker is an
    /// older build that doesn't ship the ``recovery`` blob.
    private var recoveryActions: [RecoveryAction] {
        if let raw = entry.recovery?.actions, !raw.isEmpty {
            return raw.filter { $0 != .unknown }
        }
        // Legacy broker fallback — derive from what's locally observable.
        var fallback: [RecoveryAction] = []
        let hasText = (entry.transcript?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false)
            || (entry.rawTranscript?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false)
        if let failure = entry.failureReason?.trimmingCharacters(in: .whitespacesAndNewlines), !failure.isEmpty {
            switch failure {
            case "paste_failed", "undo_safe_paste_failed":
                if hasText { fallback.append(.insertAgain) }
                if hasText { fallback.append(.copyTranscript) }
            case "ax_permission_missing":
                fallback.append(.grantAccessibility)
                if hasText { fallback.append(.copyTranscript) }
            case "no_active_text_field", "paste_kind_none_with_text":
                if hasText { fallback.append(.copyTranscript) }
            default:
                if failure.hasPrefix("capability_blocked") {
                    fallback.append(.allowApp)
                }
                if hasText { fallback.append(.copyTranscript) }
            }
        }
        if entry.replayAvailable == true {
            fallback.append(.replayAudio)
            fallback.append(.rerunInMode)
        }
        return fallback
    }

    private func primaryRecoveryButton(for actions: [RecoveryAction]) -> RecoveryStripSummary.PrimaryButton? {
        // Pick the most actionable button for the strip. Order matches a
        // user's likely intent: "fix the permission" beats "give me back
        // my text" beats "let me copy it".
        if actions.contains(.grantAccessibility) {
            return .init(title: "Grant Accessibility", isInFlight: false) {
                JunoSystemSettingsLinks.openAccessibilityPrivacy()
            }
        }
        if actions.contains(.allowApp) {
            return .init(title: "Allow this app", isInFlight: false) {
                windowNav.section = .personalization
            }
        }
        if actions.contains(.insertAgain) {
            return .init(title: isInsertingAgain ? "Inserting…" : "Insert again",
                         isInFlight: isInsertingAgain) {
                runInsertAgain()
            }
        }
        if actions.contains(.copyTranscript) {
            return .init(title: copiedFromBar ? "Copied" : "Copy transcript",
                         isInFlight: false) {
                copyTranscriptFromBar()
            }
        }
        return nil
    }

    // MARK: - Insert again handler
    //
    // Asks the broker for the saved transcript text (it has the
    // authoritative copy in SQLite, including raw vs final), writes it to
    // the clipboard and triggers paste via the existing
    // ``Clipboard.pasteAtCursor`` capability path. We deliberately do not
    // paste from Python — keystroke synthesis stays on the macOS side so
    // we get the existing focus-drift diagnostics + AX permission flow.
    private func runInsertAgain() {
        guard !isInsertingAgain else { return }
        // Optimistic local fallback: if we already have the text on the
        // entry, skip the round-trip.
        let localText = (entry.transcript?.trimmingCharacters(in: .whitespacesAndNewlines))
            ?? (entry.rawTranscript?.trimmingCharacters(in: .whitespacesAndNewlines))
            ?? ""
        if !localText.isEmpty {
            performInsertAgain(text: localText)
            return
        }
        isInsertingAgain = true
        insertAgainError = nil
        insertAgainSuccess = nil
        JunoBroker.postHistoryInsertAgain(utteranceId: entry.utteranceId) { result in
            DispatchQueue.main.async {
                isInsertingAgain = false
                switch result {
                case .success(let obj):
                    let text = (obj["text"] as? String)?
                        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                    guard !text.isEmpty else {
                        insertAgainError = "There’s no saved text to re-insert."
                        return
                    }
                    performInsertAgain(text: text)
                case .failure(let err):
                    insertAgainError = err.localizedDescription
                }
            }
        }
    }

    private func performInsertAgain(text: String) {
        // Pre-flight: AX permission gates synthesised Cmd+V. Without it
        // we'd silently no-op into Chrome / the focused app.
        if !JunoLocalCapability.processHasAccessibilityTrust() {
            insertAgainError = "Accessibility permission is off — Juno can’t paste into other apps until it’s granted."
            return
        }
        Clipboard.writeString(text)
        // Defer the paste keystroke a beat so the user has a chance to
        // refocus a target app if Juno's window is forward. Paste at
        // cursor handles the focus drift diagnostic itself.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.20) {
            let ok = Clipboard.pasteAtCursor()
            DispatchQueue.main.async {
                if ok {
                    insertAgainError = nil
                    insertAgainSuccess = "The transcript was pasted at your cursor."
                } else {
                    insertAgainError = "Paste did not succeed. The text is on your clipboard — paste it manually with Cmd+V."
                }
            }
        }
    }

    private func copyTranscriptFromBar() {
        guard let text = transcriptForCopy else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        copiedFromBar = true
        copiedFromBarResetTask?.cancel()
        copiedFromBarResetTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            guard !Task.isCancelled else { return }
            copiedFromBar = false
        }
    }

    // MARK: - Diagnostics disclosure
    //
    // Collapsed by default. Engineers / support sessions get one click to
    // a copyable bundle of failure code, app context, mode, timing —
    // exactly the things that should NEVER appear in user-facing copy.
    @ViewBuilder
    private var diagnosticsDisclosure: some View {
        let lines = diagnosticsLines
        if !lines.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Button {
                    withAnimation(.easeOut(duration: 0.18)) { showDiagnostics.toggle() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: showDiagnostics ? "chevron.down" : "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                        Text(showDiagnostics ? "Hide diagnostics" : "Show diagnostics")
                            .font(.system(size: 11, weight: .medium, design: .rounded))
                        Spacer(minLength: 0)
                    }
                    .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.7))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .junoNoFocusRing()

                if showDiagnostics {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
                                Text(line.label)
                                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.8))
                                    .frame(width: 110, alignment: .leading)
                                Text(line.value)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(JunoTheme.primaryText(scheme).opacity(0.85))
                                    .textSelection(.enabled)
                                Spacer(minLength: 0)
                            }
                        }
                        Button {
                            copyDiagnosticsToPasteboard(lines: lines)
                        } label: {
                            Label("Copy diagnostics", systemImage: "doc.on.doc")
                                .font(.system(size: 11, weight: .medium, design: .rounded))
                        }
                        .buttonStyle(.plain)
                        .junoNoFocusRing()
                        .foregroundStyle(JunoDesignTokens.accent)
                        .padding(.top, 4)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(JunoTheme.secondaryText(scheme).opacity(scheme == .dark ? 0.06 : 0.04))
                    )
                }
            }
        }
    }

    private struct DiagnosticsLine { let label: String; let value: String }

    private var diagnosticsLines: [DiagnosticsLine] {
        var out: [DiagnosticsLine] = [
            DiagnosticsLine(label: "Utterance",  value: entry.utteranceId)
        ]
        if let m = entry.mode, !m.isEmpty {
            out.append(DiagnosticsLine(label: "Mode", value: m))
        }
        if let bid = entry.context?.appBundleId, !bid.isEmpty {
            out.append(DiagnosticsLine(label: "App bundle", value: bid))
        }
        if let win = entry.context?.windowTitle, !win.isEmpty {
            out.append(DiagnosticsLine(label: "Window", value: win))
        }
        if let ms = entry.processingMs, ms > 0 {
            out.append(DiagnosticsLine(label: "Processing",
                                       value: String(format: "%.0f ms", ms)))
        }
        if let words = entry.words, words > 0 {
            out.append(DiagnosticsLine(label: "Words", value: String(words)))
        }
        if let raw = entry.failureReason?.trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty {
            out.append(DiagnosticsLine(label: "Failure code", value: raw))
        }
        if let category = entry.recovery?.category, !category.isEmpty {
            out.append(DiagnosticsLine(label: "Category", value: category))
        }
        if entry.replayAvailable == true {
            out.append(DiagnosticsLine(label: "Audio", value: "retained"))
        } else {
            out.append(DiagnosticsLine(label: "Audio", value: "not retained"))
        }
        return out
    }

    private func copyDiagnosticsToPasteboard(lines: [DiagnosticsLine]) {
        let blob = lines.map { "\($0.label): \($0.value)" }.joined(separator: "\n")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(blob, forType: .string)
    }

    private func copyTranscriptToPasteboard() {
        guard let t = transcriptForCopy else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(t, forType: .string)
        copiedTranscript = true
        copiedTranscriptResetTask?.cancel()
        copiedTranscriptResetTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_100_000_000)
            guard !Task.isCancelled else { return }
            copiedTranscript = false
        }
    }

    // MARK: - Disclosure rows
    //
    // For action entries the transcript and rewrite-diff are demoted —
    // the user came here to see the saved artifact, not to re-read what
    // they said. These quiet text-link rows live below the hero card and
    // expand inline when tapped.

    private var transcriptDisclosureRow: some View {
        Button {
            withAnimation(.easeOut(duration: 0.18)) { showTranscriptInline.toggle() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: showTranscriptInline ? "chevron.down" : "chevron.right")
                    .font(.system(size: 9, weight: .semibold))
                Text(showTranscriptInline ? "Hide what you said" : "Show what you said  ·  teach a term")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                Spacer(minLength: 0)
            }
            .foregroundStyle(JunoTheme.secondaryText(scheme))
            .contentShape(Rectangle())
            .padding(.vertical, 4)
        }
        .buttonStyle(.plain)
        .junoNoFocusRing()
    }

    private var rewriteDisclosureRow: some View {
        Button {
            withAnimation(.easeOut(duration: 0.18)) { showRewriteInline.toggle() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: showRewriteInline ? "chevron.down" : "chevron.right")
                    .font(.system(size: 9, weight: .semibold))
                Image(systemName: "sparkles")
                    .font(.system(size: 10, weight: .semibold))
                Text(showRewriteInline ? "Hide rewrite" : "Juno changed what you said")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                Spacer(minLength: 0)
            }
            .foregroundStyle(JunoDesignTokens.accent)
            .contentShape(Rectangle())
            .padding(.vertical, 4)
        }
        .buttonStyle(.plain)
        .junoNoFocusRing()
    }

    // MARK: - Footer (active actions on this entry)
    //
    // Replay audio and Re-run in another mode — quiet text links at the
    // very bottom of the page. Visible but unobtrusive. Disabled when
    // audio retention is off, with a tooltip explaining why.

    private var footerActions: some View {
        let replayOn = entry.replayAvailable == true
        let replayDisabledHelp = "Re-run and replay need the original audio. Turn on audio retention in Settings → Storage to enable them for new sessions."
        return HStack(spacing: 18) {
            if !failedActionResults.isEmpty {
                footerLink(
                    title: isRetryingActions
                        ? "Retrying action…"
                        : (failedActionResults.count == 1 ? "Retry action" : "Retry failed actions"),
                    systemImage: "arrow.clockwise",
                    help: "Check permissions and retry the failed voice action.",
                    isEnabled: !isRetryingActions
                ) {
                    retryFailedActions()
                }
            }

            footerLink(
                title: audioPlayer?.isPlaying == true ? "Stop replay" : "Replay audio",
                systemImage: audioPlayer?.isPlaying == true ? "stop.fill" : "play.fill",
                help: replayOn ? "Replay the original audio" : replayDisabledHelp,
                isEnabled: replayOn
            ) {
                if audioPlayer?.isPlaying == true { audioPlayer?.stop() }
                else { replayAudio() }
            }

            if entry.isActionHistoryRow {
                footerLink(
                    title: "Teach Juno",
                    systemImage: "plus",
                    help: "Save a name, product, acronym, or term from this action.",
                    isEnabled: true
                ) {
                    showSavePhrasePopover = true
                }
                .popover(isPresented: $showSavePhrasePopover, arrowEdge: .top) {
                    savePhrasePopoverContent
                }
            } else {
                footerLink(
                    title: "Re-run with another style",
                    systemImage: "arrow.triangle.2.circlepath",
                    help: replayOn ? "Re-run this session in a different writing style" : replayDisabledHelp,
                    isEnabled: replayOn && !isReprocessing
                ) {
                    loadModesForReprocess()
                    showReprocessSheet = true
                }
            }

            Spacer(minLength: 0)
        }
        .padding(.top, 4)
    }

    private var failedActionResults: [JunoActionResult] {
        displayedActionResults.filter { result in
            switch result.status {
            case .permissionDenied, .blockedNoPermission, .blockedToggleOff,
                 .sinkError, .timeParseFailed:
                return true
            case .ok, .pending:
                return false
            }
        }
    }

    private func footerLink(
        title: String,
        systemImage: String,
        help: String,
        isEnabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button {
            guard isEnabled else { return }
            action()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: systemImage)
                    .font(.system(size: 10, weight: .semibold))
                Text(title)
                    .font(.system(size: 12, weight: .medium, design: .rounded))
            }
            .foregroundStyle(isEnabled ? JunoTheme.secondaryText(scheme) : JunoTheme.secondaryText(scheme).opacity(0.4))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .junoNoFocusRing()
        .help(help)
    }

    private func retryFailedActions() {
        let failed = failedActionResults
        let retryRequests = failed.compactMap(Self.retryRequest(from:))
        guard !retryRequests.isEmpty else {
            retryActionError = "This action cannot be retried from History."
            return
        }
        isRetryingActions = true
        retryActionResult = nil
        retryActionError = nil
        ensureActionPermissions(for: retryRequests.map { $0.kind }) { granted, message in
            guard granted else {
                isRetryingActions = false
                retryActionError = message ?? "Permission is still missing."
                return
            }
            JunoActionExecutor.shared.execute(
                utteranceId: entry.utteranceId,
                actions: retryRequests,
                postResults: { uid, retryResults in
                    let merged = Self.mergeActionResults(
                        existing: entry.actions ?? [],
                        retryResults: retryResults
                    )
                    JunoActionExecutor.postResultsToBroker(
                        utteranceId: uid,
                        results: merged
                    )
                },
                completion: { retryResults in
                    isRetryingActions = false
                    let summary = JunoActionBatchFormatter.summarize(retryResults)
                    if retryResults.allSatisfy({ $0.status == .ok }) {
                        retryActionResult = summary.oneLine
                        retryActionError = nil
                    } else {
                        retryActionResult = nil
                        retryActionError = summary.oneLine
                    }
                }
            )
        }
    }

    private func ensureActionPermissions(
        for kinds: [JunoActionKind],
        completion: @escaping (Bool, String?) -> Void
    ) {
        var descriptors: [JunoActionPermissionDescriptor] = []
        for kind in kinds {
            let descriptor = kind.descriptor.permission
            if !descriptors.contains(descriptor) {
                descriptors.append(descriptor)
            }
        }
        ensureActionPermissions(descriptors, completion: completion)
    }

    private func ensureActionPermissions(
        _ descriptors: [JunoActionPermissionDescriptor],
        completion: @escaping (Bool, String?) -> Void
    ) {
        guard let descriptor = descriptors.first else {
            completion(true, nil)
            return
        }
        let remaining = Array(descriptors.dropFirst())
        ensureActionPermission(descriptor) { granted, message in
            guard granted else {
                completion(false, message)
                return
            }
            ensureActionPermissions(remaining, completion: completion)
        }
    }

    private func ensureActionPermission(
        _ descriptor: JunoActionPermissionDescriptor,
        completion: @escaping (Bool, String?) -> Void
    ) {
        let store = JunoActionPermissionStore.shared
        store.refreshAll(forceNotesProbe: false)
        if store.status(for: descriptor).isGranted {
            completion(true, nil)
            return
        }

        let finish: (JunoActionPermissionStatus) -> Void = { status in
            if status.isGranted {
                completion(true, nil)
                return
            }
            switch descriptor {
            case .reminders:
                store.openRemindersSettings()
                completion(false, "Reminders permission is still off. Enable it in System Settings, then retry again.")
            case .calendarEvents:
                store.openCalendarSettings()
                completion(false, "Calendar permission is still off. Enable it in System Settings, then retry again.")
            case .notesAutomation:
                store.openAutomationSettings()
                completion(false, "Notes Automation permission is still off. Enable Notes for Juno, then retry again.")
            }
        }

        switch descriptor {
        case .reminders:
            store.requestReminders(finish)
        case .calendarEvents:
            store.requestCalendarEvents(finish)
        case .notesAutomation:
            store.requestNotesAutomation(finish)
        }
    }

    private static func retryRequest(from result: JunoActionResult) -> JunoActionRequest? {
        let body = (result.body ?? result.bodyPreview).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !body.isEmpty || result.kind == .alarm else { return nil }
        let when = result.whenIso.map {
            JunoParsedTime(iso: $0, confidence: 1.0, source: "history_retry")
        }
        return JunoActionRequest(
            junoId: result.junoId,
            sinkId: result.sinkId,
            kind: result.kind,
            body: body.isEmpty ? result.kind.descriptor.displayName : body,
            rawSpan: body.isEmpty ? result.kind.descriptor.displayName : body,
            when: when,
            operation: result.operation ?? .create
        )
    }

    private static func mergeActionResults(
        existing: [JunoActionResult],
        retryResults: [JunoActionResult]
    ) -> [JunoActionResult] {
        var unused = retryResults
        var merged = existing.map { current -> JunoActionResult in
            guard let idx = unused.firstIndex(where: { $0.junoId == current.junoId }) else {
                return current
            }
            return unused.remove(at: idx)
        }
        merged.append(contentsOf: unused)
        return merged
    }

    // MARK: - Transcript card

    /// The transcript card. For pure-dictation entries (no actions) this
    /// is the page's hero — bigger typography, generous line spacing, no
    /// height cap. For action entries it appears collapsed below the
    /// hero card and expands inline via the disclosure row above.
    ///
    /// The Teach-a-term button lives at the bottom as a labeled button —
    /// previously a cryptic `+` icon in the header, now an obvious
    /// affordance the user can read.
    private func transcriptCard(_ text: String) -> some View {
        historyDetailCard(padding: 16) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .center, spacing: 8) {
                    Text("TRANSCRIPT")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .tracking(1.1)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                    Spacer(minLength: 0)
                    if let wc = entry.wordCountLabel {
                        Text(wc)
                            .font(.system(size: 11, weight: .medium, design: .rounded))
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                    }
                }

                transcriptBody(text)

                HStack(spacing: 10) {
                    Button {
                        showSavePhrasePopover = true
                    } label: {
                        Label("Teach Juno", systemImage: "plus")
                            .font(.system(size: 12, weight: .medium, design: .rounded))
                    }
                    .junoSecondaryActionButton()
                    .popover(isPresented: $showSavePhrasePopover, arrowEdge: .top) {
                        savePhrasePopoverContent
                    }
                    Spacer(minLength: 0)
                }
            }
        }
    }

    /// One card, two labelled sections: the final transcript (hero) and,
    /// when Juno rewrote anything, the user's original phrasing below a
    /// divider. Replaces the previous render of two separately-styled
    /// slabs which felt visually disconnected and dumped two walls of
    /// near-identical text for a single utterance.
    ///
    /// Reuses every existing primitive: ``historyDetailCard`` chrome, the
    /// canonical TRANSCRIPT/word-count eyebrow, ``transcriptBody`` chunked
    /// renderer, the existing Save Phrase button + popover, and the
    /// sparkles/"YOU SAID" eyebrow that previously lived inside
    /// ``junoContributionCard``.
    private func unifiedDictationDetailCard(final: String, raw: String?) -> some View {
        let rawTrimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let hasRewrite = !rawTrimmed.isEmpty
        let isLong = entry.transcriptBodyHeightCap >= 280
        let stateChipText = heroStateChip
        return historyDetailCard(padding: 16) {
            VStack(alignment: .leading, spacing: 14) {
                // Eyebrow: "PASTED TEXT · 64 words" + optional state
                // chip ("VERBATIM MODE" / "NO REWRITE"). Reads
                // immediately as "what's the artifact and what state
                // is it in" — without making the user scan to the
                // bottom of the card to figure it out.
                heroEyebrow(
                    label: heroEyebrowLabel,
                    wordCount: entry.wordCountLabel,
                    stateChip: stateChipText,
                    showsExpandToggle: isLong
                )

                transcriptBodyConstrained(final)

                // Hero actions row — Copy is the primary action (this
                // is what the user came here to do), Teach is the
                // secondary nudge, and "View full ↗" only appears for
                // long transcripts whose body is currently capped.
                HStack(spacing: 10) {
                    Button {
                        copyTranscriptFromHero(final)
                    } label: {
                        Label(
                            copiedFromHero ? "Copied" : "Copy",
                            systemImage: copiedFromHero ? "checkmark" : "doc.on.doc"
                        )
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                    }
                    .junoPrimaryActionButton()

                    Button {
                        showSavePhrasePopover = true
                    } label: {
                        Label("Teach Juno", systemImage: "plus")
                            .font(.system(size: 12, weight: .medium, design: .rounded))
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                    .junoSecondaryActionButton()
                    .popover(isPresented: $showSavePhrasePopover, arrowEdge: .top) {
                        savePhrasePopoverContent
                    }

                    Spacer(minLength: 0)

                    if isLong {
                        heroViewFullToggle
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                }

                if hasRewrite {
                    Divider().opacity(0.5)
                    // Collapsed by default — the user came to read the
                    // pasted text, not the original. The disclosure
                    // surfaces the edit count in the header so they
                    // know there's something to look at if they want.
                    provenanceSection(raw: rawTrimmed, final: final, defaultExpanded: false)
                }
            }
        }
    }

    /// Eyebrow row for the dictation hero. Mirrors the existing
    /// monospaced uppercase eyebrow used elsewhere in the app
    /// (Re-run card, transcriptCard) so the new card stays in the
    /// shared visual rhythm. The state chip surfaces the per-utterance
    /// mode in one place so the user knows *why* the body looks the
    /// way it does without having to dig into Diagnostics.
    @ViewBuilder
    private func heroEyebrow(
        label: String,
        wordCount: String?,
        stateChip: String?,
        showsExpandToggle: Bool
    ) -> some View {
        HStack(alignment: .center, spacing: 8) {
            Text(label)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .tracking(1.1)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
            if let wordCount {
                Text("· \(wordCount)")
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }
            if let stateChip {
                Text(stateChip)
                    .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                    .tracking(0.9)
                    .foregroundStyle(JunoDesignTokens.accent)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 2)
                    .background(
                        Capsule()
                            .fill(JunoDesignTokens.accent.opacity(0.12))
                    )
                    .padding(.leading, 4)
            }
            Spacer(minLength: 0)
        }
    }

    /// Hero label per entry shape. Pure dictation reads as "PASTED
    /// TEXT" because that's what landed in the app; failure paths
    /// where there's no rewrite output use "YOUR TRANSCRIPT" because
    /// nothing was pasted to call "pasted text".
    private var heroEyebrowLabel: String {
        if entry.displayFailureReason != nil && entry.transcript == nil {
            return "YOUR TRANSCRIPT"
        }
        return "PASTED TEXT"
    }

    /// State chip text. Returns nil when no chip should render — the
    /// happy path (rewritten) is the default and doesn't need a
    /// label. Verbatim mode is the only mode that semantically
    /// implies the body is "as spoken" rather than cleaned.
    private var heroStateChip: String? {
        let mode = (entry.mode ?? "").lowercased()
        if mode.contains("verbatim") {
            return "VERBATIM"
        }
        return nil
    }

    /// "View full ↗" / "Collapse ↘" toggle in the eyebrow / hero
    /// actions row. Visible only when the body is currently capped
    /// (long transcript); toggling it removes the 320pt cap so the
    /// body expands to its full height. The outer page ScrollView
    /// still handles overall page scroll.
    private var heroViewFullToggle: some View {
        Button {
            withAnimation(.easeOut(duration: 0.18)) {
                heroExpandedFull.toggle()
            }
        } label: {
            HStack(spacing: 4) {
                Text(heroExpandedFull ? "Collapse" : "View full")
                    .font(.system(size: 11.5, weight: .medium, design: .rounded))
                Image(systemName: heroExpandedFull ? "chevron.up.chevron.down" : "arrow.up.right")
                    .font(.system(size: 9, weight: .semibold))
            }
            .foregroundStyle(JunoDesignTokens.accent)
        }
        .buttonStyle(.plain)
        .junoNoFocusRing()
    }

    /// Render the transcript body with a fixed height cap so the
    /// "Copy" + "Teach Juno a term" buttons immediately below the body
    /// stay visible on screen no matter how long the transcript is.
    /// The body scrolls inside the cap; the page never has to scroll
    /// past a wall of text to find the actions. When the user taps
    /// "View full ↗" in the eyebrow row the cap is lifted (the outer
    /// page scroll handles overall page length).
    ///
    /// Short transcripts (one or two lines) take the SwiftUI intrinsic
    /// height inside the ScrollView, which sizes to content up to the
    /// max — so there's no awkward empty space for tiny dictations.
    @ViewBuilder
    private func transcriptBodyConstrained(_ text: String) -> some View {
        if heroExpandedFull {
            transcriptBody(text)
        } else {
            ScrollView(.vertical, showsIndicators: true) {
                transcriptBody(text)
                    .padding(.trailing, 4)
            }
            .frame(maxHeight: 320)
        }
    }

    /// In-hero Copy. Mirrors the overflow-menu Copy and the sticky
    /// action bar Copy — three call sites, all routing to the same
    /// pasteboard write — so the user can copy from wherever their
    /// eye lands without thinking about it. Feedback ("Copied")
    /// lives in this specific button only; the other locations use
    /// their own flags so visual feedback doesn't bleed across.
    private func copyTranscriptFromHero(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        copiedFromHero = true
        copiedFromHeroResetTask?.cancel()
        copiedFromHeroResetTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_300_000_000)
            guard !Task.isCancelled else { return }
            copiedFromHero = false
        }
    }

    /// "What you said" provenance section. Renders the disclosure
    /// header (chevron + label + edit-count legend) and, when
    /// expanded, the inline word-level diff between raw and final.
    /// One component for both dictation and action entries so the
    /// reveal looks identical regardless of entry type.
    ///
    /// ``defaultExpanded`` is the open-by-default policy:
    ///   * dictation with rewrite → true (user came here for it)
    ///   * action with transcript → false (artifact is the hero)
    /// The user's manual toggle wins after they touch it; we only
    /// apply the default once per entry (tracked via
    /// ``provenanceDefaultedFor``) so re-renders don't pop it back
    /// open after the user collapses it.
    @ViewBuilder
    private func provenanceSection(raw: String, final: String, defaultExpanded: Bool) -> some View {
        let rawTrimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let finalTrimmed = final.trimmingCharacters(in: .whitespacesAndNewlines)
        let report = JunoHistoryDiff.polishReport(raw: raw, final: final)
        let entryKey = entry.id
        if report.counts.hasAny {
            VStack(alignment: .leading, spacing: 12) {
                Button {
                    withAnimation(.easeOut(duration: 0.18)) {
                        showProvenance.toggle()
                    }
                } label: {
                    HStack(spacing: 8) {
                        Text("JUNO'S POLISH")
                            .junoType(.eyebrow)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                        Spacer(minLength: 8)
                        Image(systemName: showProvenance ? "chevron.up" : "chevron.down")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                    }
                    .contentShape(Rectangle())
                    .padding(.vertical, 4)
                }
                .buttonStyle(.plain)
                .junoNoFocusRing()

                if showProvenance {
                    VStack(alignment: .leading, spacing: 12) {
                        polishSummaryChips(report.counts)
                        Rectangle()
                            .fill(JunoUI.hairline(.regular, scheme: scheme))
                            .frame(height: 0.5)
                        JunoHistoryDiff.renderInlineDiff(segments: report.segments, scheme: scheme)
                    }
                    .transition(.opacity)
                }
            }
            .onAppear {
                if provenanceDefaultedFor != entryKey {
                    provenanceDefaultedFor = entryKey
                    showProvenance = defaultExpanded
                }
            }
            .onChange(of: entry.id) { _ in
                provenanceDefaultedFor = entryKey
                showProvenance = defaultExpanded
            }
        }
        else if !rawTrimmed.isEmpty && !finalTrimmed.isEmpty && rawTrimmed != finalTrimmed {
            VStack(alignment: .leading, spacing: 12) {
                Button {
                    withAnimation(.easeOut(duration: 0.18)) {
                        showProvenance.toggle()
                    }
                } label: {
                    HStack(spacing: 8) {
                        Text("WHAT YOU SAID")
                            .junoType(.eyebrow)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                        Spacer(minLength: 8)
                        Image(systemName: showProvenance ? "chevron.up" : "chevron.down")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                    }
                    .contentShape(Rectangle())
                    .padding(.vertical, 4)
                }
                .buttonStyle(.plain)
                .junoNoFocusRing()

                if showProvenance {
                    transcriptBody(rawTrimmed)
                        .font(.system(size: 14, design: .rounded))
                        .lineSpacing(4)
                        .transition(.opacity)
                }
            }
            .onAppear {
                if provenanceDefaultedFor != entryKey {
                    provenanceDefaultedFor = entryKey
                    showProvenance = defaultExpanded
                }
            }
            .onChange(of: entry.id) { _ in
                provenanceDefaultedFor = entryKey
                showProvenance = defaultExpanded
            }
        }
    }

    /// Compact summary chips: "12 punctuation · 5 caps · 3 word swaps · 3 fillers".
    /// Sits above the inline diff so the user gets a quantified read at a glance,
    /// then can scan the diff below for the actual word-by-word changes.
    private func polishSummaryChips(_ counts: JunoHistoryDiff.PolishCounts) -> some View {
        let parts: [String] = [
            counts.punctuation     > 0 ? "\(counts.punctuation) punctuation"        : "",
            counts.capitalizations > 0 ? "\(counts.capitalizations) capitalizations": "",
            counts.wordSwaps       > 0 ? "\(counts.wordSwaps) word swap\(counts.wordSwaps == 1 ? "" : "s")" : "",
            counts.fillersRemoved  > 0 ? "\(counts.fillersRemoved) filler\(counts.fillersRemoved == 1 ? "" : "s") removed" : "",
        ].filter { !$0.isEmpty }

        return Text(parts.joined(separator: "  ·  "))
            .font(.system(size: 11, weight: .semibold, design: .monospaced))
            .tracking(0.4)
            .foregroundStyle(JunoTheme.secondaryText(scheme))
            .fixedSize(horizontal: false, vertical: true)
    }

    /// Render the transcript body. For short transcripts we keep a single
    /// :class:`Text` view (cheap, full ``.textSelection`` continuity). For
    /// long transcripts we split on paragraph boundaries and render each
    /// paragraph as its own ``Text`` inside a ``VStack`` — this prevents
    /// SwiftUI from synchronously laying out a multi-thousand-word block,
    /// which was both slow and let the selectable Text intercept trackpad
    /// scroll over the body. Selection still spans paragraphs.
    ///
    /// The threshold is character-based, not word-based, so a transcript
    /// of one giant unbroken sentence (no paragraph breaks) still falls
    /// into the chunked path and is split on sentence boundaries.
    @ViewBuilder
    private func transcriptBody(_ text: String) -> some View {
        let chunks = Self.chunkTranscript(text)
        if chunks.count <= 1 {
            Text(text)
                .font(.system(size: 14, weight: .regular, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .textSelection(.enabled)
                .lineSpacing(2.5)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(Array(chunks.enumerated()), id: \.offset) { _, chunk in
                    Text(chunk)
                        .font(.system(size: 14, weight: .regular, design: .rounded))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                        .textSelection(.enabled)
                        .lineSpacing(2.5)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    /// Threshold above which we switch to chunked rendering. Empirical:
    /// SwiftUI ``Text`` with ``.textSelection`` stays smooth up to ~1.5 KB
    /// in this layout; we leave headroom.
    private static let transcriptChunkingThresholdChars = 1_200

    /// Split a transcript into render-friendly paragraphs. Falls back to
    /// sentence-boundary splits when the source has no blank lines (the
    /// common dictation case is one wall of text).
    static func chunkTranscript(_ text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count <= transcriptChunkingThresholdChars {
            return [trimmed]
        }
        let byParagraph = trimmed
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if byParagraph.count > 1 {
            return byParagraph
        }
        // No paragraph breaks: split on sentence terminators while keeping
        // them on the preceding chunk. Cap each chunk at a soft size so a
        // pathological transcript (no punctuation either) still chunks.
        let softCap = 600
        var chunks: [String] = []
        var current = ""
        var i = trimmed.startIndex
        while i < trimmed.endIndex {
            let ch = trimmed[i]
            current.append(ch)
            let isTerminator = (ch == "." || ch == "?" || ch == "!" || ch == "\n")
            let nextIsSpace: Bool = {
                let next = trimmed.index(after: i)
                guard next < trimmed.endIndex else { return true }
                return trimmed[next].isWhitespace
            }()
            if (isTerminator && nextIsSpace && current.count >= 200) || current.count >= softCap {
                chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                current = ""
            }
            i = trimmed.index(after: i)
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty {
            chunks.append(tail)
        }
        return chunks.isEmpty ? [trimmed] : chunks
    }

    // MARK: - Action hero card
    //
    // For action entries this is the page's hero: one card per action
    // that looks like a real preview tile of where the artifact landed
    // (Notes / Reminders / Calendar). Replaces the old "What Juno did"
    // wrapper card and inner action rows. Design rationale:
    //
    // - Destination line ("Notes · Juno folder") replaces the
    //   "Saved" status pill — destination *is* the success signal, and
    //   presence of this card on a History page already implies success.
    //   Status chips only appear on failure.
    // - Body has its own row with no horizontal competition, so it can
    //   wrap to up to four readable lines without overlap.
    // - Primary action button ("Open in Notes") sits on its own row at
    //   the bottom, right-aligned, filled accent style — the only filled
    //   button on the page.
    // - On failure, the card flips: destination becomes the failure
    //   reason and the primary button becomes a recovery action where
    //   one exists (e.g. Open System Settings for permission denied).

    @ViewBuilder
    private func actionHeroSection(actions: [JunoActionResult]) -> some View {
        VStack(spacing: 12) {
            ForEach(Array(actions.enumerated()), id: \.offset) { _, action in
                actionHeroCard(action)
            }
        }
    }

    @ViewBuilder
    private func actionHeroCard(_ action: JunoActionResult) -> some View {
        let descriptor = action.kind.descriptor
        let isSuccess = action.status == .ok
        let isPending = action.status == .pending
        let tint: Color = {
            switch action.status {
            case .ok, .pending: return descriptor.accent
            case .permissionDenied, .blockedNoPermission, .blockedToggleOff: return .orange
            case .sinkError, .timeParseFailed: return .red
            }
        }()

        VStack(alignment: .leading, spacing: 14) {
            // Top row: kind glyph + destination / failure line.
            HStack(alignment: .center, spacing: 12) {
                JunoActionNativeIconTile(kind: action.kind, tileSize: 36, iconSize: 32, fallbackTint: tint)

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(heroPrimaryLine(for: action))
                            .font(.system(size: 14, weight: .semibold, design: .rounded))
                            .foregroundStyle(JunoTheme.primaryText(scheme))
                            .lineLimit(1)
                            .truncationMode(.tail)
                        if action.kind == .alarm {
                            JunoAlarmInfoButton()
                        }
                    }
                    if let secondary = heroSecondaryLine(for: action) {
                        Text(secondary)
                            .font(.system(size: 12, weight: .regular, design: .rounded))
                            .foregroundStyle(isSuccess || isPending ? JunoTheme.secondaryText(scheme) : tint)
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 0)
                if isPending {
                    ProgressView().controlSize(.small)
                }
            }

            // Body preview — own row, no horizontal competition.
            Text(action.bodyPreview.isEmpty ? "(empty)" : action.bodyPreview)
                .font(.system(size: 14, weight: .regular, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .lineSpacing(2)
                .lineLimit(6)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Bottom row: primary recovery / open button on the right.
            if !isPending, let primary = heroPrimaryButton(for: action) {
                HStack {
                    Spacer(minLength: 0)
                    primary
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(JunoTheme.cardBackground(scheme))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(tint.opacity(scheme == .dark ? 0.32 : 0.18), lineWidth: 0.7)
        )
    }

    /// Top line of the hero — the destination on success, the failure
    /// reason on failure. Reads as "Notes · Juno folder",
    /// "Reminders · Tomorrow at 9:00 AM", "Couldn't save to Notes",
    /// "Permission needed", etc.
    private func heroPrimaryLine(for action: JunoActionResult) -> String {
        switch action.status {
        case .ok, .pending:
            switch action.kind {
            case .note: return "Notes  ·  \(JunoNotesFolderName) folder"
            case .reminder: return "Reminders"
            case .alarm: return "Alarm"
            }
        case .permissionDenied, .blockedNoPermission:
            return "Permission needed"
        case .blockedToggleOff:
            return "Voice Actions are off"
        case .sinkError:
            switch action.kind {
            case .note:     return "Couldn't save to Notes"
            case .reminder: return "Couldn't save to Reminders"
            case .alarm:    return "Couldn't save alarm"
            }
        case .timeParseFailed:
            return "Couldn't read the time"
        }
    }

    /// Secondary line under the destination: parsed time on success,
    /// error detail on failure.
    private func heroSecondaryLine(for action: JunoActionResult) -> String? {
        if let iso = action.whenIso, let formatted = formatActionDue(iso) {
            return formatted
        }
        if action.status == .pending {
            return "Saving…"
        }
        if action.status != .ok, let err = action.error, !err.isEmpty {
            return err
        }
        return nil
    }

    /// The single primary button on the hero card. Open-the-artifact for
    /// success; recovery (Open System Settings, Open Voice Actions) for
    /// blocked states. `nil` when no useful action exists.
    private func heroPrimaryButton(for action: JunoActionResult) -> AnyView? {
        switch action.status {
        case .ok:
            guard let url = actionDeepLink(action) else { return nil }
            return AnyView(
                Button {
                    NSWorkspace.shared.open(url)
                } label: {
                    Label(deepLinkLabel(for: action), systemImage: deepLinkIcon(for: action))
                        .font(.system(size: 12, weight: .semibold))
                }
                .junoPrimaryActionButton()
            )
        case .permissionDenied, .blockedNoPermission:
            return AnyView(
                Button {
                    openPermissionSettings(for: action.kind)
                } label: {
                    Label("Open System Settings", systemImage: "arrow.up.right.square")
                        .font(.system(size: 12, weight: .semibold))
                }
                .junoPrimaryActionButton()
            )
        case .blockedToggleOff:
            return AnyView(
                Button {
                    windowNav.section = .actions
                } label: {
                    Label("Open Voice Actions", systemImage: "switch.2")
                        .font(.system(size: 12, weight: .semibold))
                }
                .junoSecondaryActionButton()
            )
        case .sinkError, .timeParseFailed, .pending:
            return nil
        }
    }

    private func openPermissionSettings(for kind: JunoActionKind) {
        switch kind {
        case .note:     JunoSystemSettingsLinks.openAutomationPrivacy()
        case .reminder: JunoSystemSettingsLinks.openRemindersPrivacy()
        case .alarm:    JunoSystemSettingsLinks.openCalendarsPrivacy()
        }
    }

    private func actionDeepLink(_ action: JunoActionResult) -> URL? {
        guard action.status == .ok, let s = action.sinkUrl, !s.isEmpty else { return nil }
        return URL(string: s)
    }

    private func deepLinkLabel(for action: JunoActionResult) -> String {
        switch action.kind {
        case .note: return "Open in Notes"
        case .reminder: return "Open in Reminders"
        case .alarm: return "Open in Calendar"
        }
    }

    private func deepLinkIcon(for action: JunoActionResult) -> String {
        switch action.kind {
        case .note: return "note.text"
        case .reminder: return "bell"
        case .alarm: return "alarm"
        }
    }

    private static let actionDueFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "EEE, MMM d 'at' h:mm a"
        return f
    }()

    private func formatActionDue(_ iso: String) -> String? {
        let p = ISO8601DateFormatter()
        p.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = p.date(from: iso) { return Self.actionDueFormatter.string(from: d) }
        p.formatOptions = [.withInternetDateTime]
        if let d = p.date(from: iso) { return Self.actionDueFormatter.string(from: d) }
        return nil
    }

    // MARK: - Juno contribution card

    /// Side-by-side rewrite diff. The `expanded` flag toggles between
    /// "show original inline" (variant A — collapsed by default in the
    /// disclosure flow) and "always show original" (variant C with a
    /// rewrite, where this card is the user's debug surface).
    private func junoContributionCard(raw: String, final: String, expanded: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Label("Juno changed what you said", systemImage: "sparkles")
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoDesignTokens.accent)
                Spacer(minLength: 0)
                if !expanded {
                    Button(showRaw ? "Hide original" : "Show original") {
                        withAnimation(.easeOut(duration: 0.18)) { showRaw.toggle() }
                    }
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 14)
            .padding(.top, 12)
            .padding(.bottom, (expanded || showRaw) ? 10 : 12)

            if expanded || showRaw {
                Divider().padding(.horizontal, 14)
                VStack(alignment: .leading, spacing: 8) {
                    Text("YOU SAID")
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .tracking(0.9)
                        .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.75))
                    Text(raw)
                        .font(.system(size: 13, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .textSelection(.enabled)
                        .lineSpacing(2)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(14)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(JunoDesignTokens.accent.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(JunoDesignTokens.accent.opacity(0.14), lineWidth: 0.5)
        )
    }

    /// Compact popover for teaching Juno a single name, term, or
    /// correction. The field starts **empty** — pre-filling with a chunk
    /// of transcript was always wrong because vocabulary entries are
    /// short atomic terms ("AcmeCorp", "QBR"), never sentences.
    /// We extract likely candidates from the transcript and offer them
    /// as one-tap chips; the user can also type their own.
    private var savePhrasePopoverContent: some View {
        // Prefer LLM-suggested candidates when the writer answers; fall
        // back to the regex heuristic so the popover is never empty if
        // the writer is unavailable / cold-starting / errors out.
        let candidates = savePhraseCandidates
        return VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Teach Juno")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Spacer()
                Button("Open Dictionary & Memory") {
                    showSavePhrasePopover = false
                    // Only forward a prefill if it's actually a term-shaped
                    // string (short, no spaces). Otherwise open empty so
                    // the next page doesn't inherit the old transcript-blob
                    // bug.
                    let trimmed = phraseDraft.trimmingCharacters(in: .whitespacesAndNewlines)
                    let prefill: String? = (trimmed.count > 0 && isLearnableTermShape(trimmed) && phraseDraftIssue == nil)
                        ? trimmed : nil
                    windowNav.openDictionaryAndMemory(categoryRaw: "vocab", vocabPrefill: prefill)
                }
                .buttonStyle(.plain)
                .junoNoFocusRing()
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(JunoDesignTokens.accent)
            }
            Text("Save a name, acronym, or jargon Juno mis-hears. One term at a time — not a whole sentence.")
                .font(.system(size: 11, weight: .regular, design: .rounded))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .fixedSize(horizontal: false, vertical: true)

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text(llmCandidates != nil ? "Juno suggests" : "From this transcript")
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .tracking(0.7)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                    if llmCandidatesLoading {
                        ProgressView().controlSize(.mini)
                        Text("reading…")
                            .font(.system(size: 9.5, design: .monospaced))
                            .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.7))
                    } else if knownVocabularyLoading {
                        ProgressView().controlSize(.mini)
                        Text("checking memory…")
                            .font(.system(size: 9.5, design: .monospaced))
                            .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.7))
                    }
                    Spacer(minLength: 0)
                }
                if !candidates.isEmpty {
                    candidateFlow(candidates)
                } else if !llmCandidatesLoading {
                    Text("Nothing obvious to suggest — type a term below.")
                        .font(.system(size: 11, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
            }

            TextField("Type a term…", text: $phraseDraft)
                .textFieldStyle(.roundedBorder)
                .focusEffectDisabled()
            if let issue = phraseDraftIssue {
                Text(issue)
                    .font(.system(size: 10.5, weight: .medium, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Button("Cancel") {
                    showSavePhrasePopover = false
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .junoNoFocusRing()
                Spacer()
                Button(isSavingPhrase ? "Saving…" : "Save") {
                    guard phraseDraftIssue == nil else { return }
                    onSavePhrase(phraseDraft)
                    showSavePhrasePopover = false
                }
                .junoPrimaryActionButton()
                .disabled(
                    isSavingPhrase ||
                    phraseDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
                    phraseDraftIssue != nil
                )
            }
        }
        .padding(16)
        .frame(width: 380)
        .onAppear {
            fetchKnownVocabularyIfNeeded()
            fetchLLMCandidatesIfNeeded()
        }
    }

    /// Ask the broker's writer-extract endpoint for vocab candidates from
    /// the current transcript. Falls through to the regex heuristic if
    /// the writer is unavailable (cold backend, network error, no
    /// extraction support). Idempotent — only fires once per popover.
    private func fetchLLMCandidatesIfNeeded() {
        guard llmCandidates == nil, !llmCandidatesLoading else { return }
        let transcript = entry.transcript?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !transcript.isEmpty else { return }
        llmCandidatesLoading = true
        JunoBroker.postJSON(
            path: "api/broker/writer/extract",
            payload: ["text": transcript, "kind": "vocab", "limit": 6]
        ) { obj in
            llmCandidatesLoading = false
            guard (obj["ok"] as? Bool) == true,
                  let arr = obj["candidates"] as? [[String: Any]] else {
                return
            }
            let terms = arr.compactMap { $0["term"] as? String }
            llmCandidates = filterLearningCandidates(terms)
        }
    }

    private func fetchKnownVocabularyIfNeeded() {
        guard !knownVocabularyLoading else { return }
        knownVocabularyLoading = true
        JunoBroker.getJSON(path: "api/broker/memory/vocab") { obj in
            knownVocabularyLoading = false
            guard (obj["ok"] as? Bool) == true else { return }
            var known = MemoryStoreViewModel.protectedVocabTerms
            for entry in (obj["entries"] as? [[String: Any]]) ?? [] {
                for key in ["term", "canonical_form"] {
                    if let raw = entry[key] as? String, let normalized = learningTermKey(raw) {
                        known.insert(normalized)
                    }
                }
                for alias in (entry["aliases"] as? [String]) ?? [] {
                    if let normalized = learningTermKey(alias) {
                        known.insert(normalized)
                    }
                }
            }
            knownVocabularyTerms = known
        }
    }

    private func filterLearningCandidates(_ rawCandidates: [String]) -> [String] {
        var seen = Set<String>()
        var filtered: [String] = []
        for raw in rawCandidates {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard isLearnableTermShape(trimmed), let key = learningTermKey(trimmed) else { continue }
            guard !MemoryStoreViewModel.protectedVocabTerms.contains(key) else { continue }
            guard !knownVocabularyTerms.contains(key) else { continue }
            guard seen.insert(key).inserted else { continue }
            filtered.append(trimmed)
            if filtered.count >= 6 { break }
        }
        return filtered
    }

    private func learningTermKey(_ raw: String) -> String? {
        // Match the Python broker's fold_key so the UI's dedup verdict
        // ("we already have this term") agrees with the server's verdict
        // ("already_known"). Without that alignment the user sees their
        // candidate chip disappear while the server has stored nothing
        // new — or vice versa.
        return JunoMemoryFold.foldKeyOrNil(raw)
    }

    private func isLearnableTermShape(_ raw: String) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 2, trimmed.count <= 60 else { return false }
        let words = trimmed.split { $0.isWhitespace }
        guard words.count >= 1, words.count <= 3 else { return false }
        return trimmed.rangeOfCharacter(from: .letters) != nil
    }

    /// Wrapping row of candidate chips. Tapping a chip fills the field —
    /// faster than copy/paste and makes the "pick a name from what you
    /// just said" path obvious.
    private func candidateFlow(_ candidates: [String]) -> some View {
        JunoFlowLayout(spacing: 6, runSpacing: 6) {
            ForEach(candidates, id: \.self) { term in
                Button {
                    phraseDraft = term
                } label: {
                    Text(term)
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .padding(.horizontal, 9)
                        .padding(.vertical, 4)
                        .background(
                            Capsule().fill(
                                phraseDraft == term
                                    ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.22 : 0.14)
                                    : JunoTheme.elevatedCard(scheme)
                            )
                        )
                        .overlay(
                            Capsule().strokeBorder(
                                phraseDraft == term
                                    ? JunoDesignTokens.accent.opacity(0.45)
                                    : JunoTheme.border(scheme).opacity(0.35),
                                lineWidth: 0.6
                            )
                        )
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                }
                .buttonStyle(.plain)
                .junoNoFocusRing()
            }
        }
    }

    // MARK: - App icon

    @ViewBuilder
    private var appIconView: some View {
        if entry.isActionHistoryRow {
            Image(systemName: "bolt.badge.checkmark")
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(JunoDesignTokens.accent)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(JunoDesignTokens.accent.opacity(scheme == .dark ? 0.18 : 0.10))
        } else {
            let bundleId = entry.context?.appBundleId
            let url  = bundleId.flatMap {
                NSWorkspace.shared.urlForApplication(withBundleIdentifier: $0)
            }
            let img  = url.map { NSWorkspace.shared.icon(forFile: $0.path) }
                ?? NSImage(systemSymbolName: "app", accessibilityDescription: nil)
                ?? NSImage()
            Image(nsImage: img)
                .resizable()
                .scaledToFit()
        }
    }

    // MARK: - Re-process

    private var reprocessSheet: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Re-run with a different style")
                        .font(.system(.title2, design: .rounded).weight(.semibold))
                        .foregroundStyle(JunoTheme.primaryText(scheme))

                    Text("Juno re-transcribes this recording using the style you pick. The original entry stays unchanged — the result appears as a preview you can copy.")
                        .font(.system(.callout, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)

                    if reprocessModeList.isEmpty {
                        ProgressView("Loading styles…")
                            .font(.system(.footnote, design: .rounded))
                            .padding(.top, 8)
                    } else {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Style")
                                .font(.caption)
                                .foregroundStyle(JunoTheme.secondaryText(scheme))

                            VStack(spacing: 0) {
                                ForEach(Array(reprocessModeList.enumerated()), id: \.element.id) { index, mode in
                                    Button {
                                        selectedReprocessMode = mode.id
                                    } label: {
                                        HStack(spacing: 10) {
                                            Image(systemName: mode.isCustom ? "slider.horizontal.3" : "sparkles")
                                                .font(.system(size: 12, weight: .semibold))
                                                .foregroundStyle(JunoDesignTokens.accent)
                                                .frame(width: 16)
                                            Text(mode.label)
                                                .font(.system(size: 13, design: .rounded))
                                                .foregroundStyle(JunoTheme.primaryText(scheme))
                                            Spacer(minLength: 0)
                                            if selectedReprocessMode == mode.id {
                                                Image(systemName: "checkmark")
                                                    .font(.system(size: 11, weight: .semibold))
                                                    .foregroundStyle(JunoDesignTokens.accent)
                                            }
                                        }
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 9)
                                        .contentShape(Rectangle())
                                    }
                                    .buttonStyle(.plain)
                                    if index < reprocessModeList.count - 1 {
                                        Divider().padding(.leading, 38)
                                    }
                                }
                            }
                            .background(
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .fill(JunoTheme.elevatedCard(scheme))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .strokeBorder(JunoTheme.border(scheme).opacity(0.4), lineWidth: 0.6)
                            )
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 400)

            Divider().opacity(0.35).padding(.top, 12)

            HStack {
                Button("Cancel") { showReprocessSheet = false }
                    .junoSecondaryActionButton()
                Spacer()
                Button("Re-run") {
                    showReprocessSheet = false
                    runReprocess()
                }
                .junoPrimaryActionButton()
                .disabled(selectedReprocessMode.isEmpty || reprocessModeList.isEmpty)
            }
            .padding(.top, 12)
        }
        .padding(18)
        .frame(minWidth: 420)
    }

    private var reprocessLoadingCard: some View {
        historyDetailCard {
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text("Re-running with \(reprocessResultModeLabel.isEmpty ? "selected style" : reprocessResultModeLabel)…")
                    .font(.system(size: 12, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func reprocessResultCard(_ text: String) -> some View {
        historyDetailCard {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .center) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(JunoDesignTokens.accent)
                    Text("RE-RUN · \(reprocessResultModeLabel.uppercased())")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .tracking(1.0)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                    Spacer(minLength: 0)
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(text, forType: .string)
                    } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                    }
                    .junoSecondaryActionButton()
                }
                Text(text)
                    .font(.system(size: 12.5, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .textSelection(.enabled)
                    .lineSpacing(1.5)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .transition(.opacity.combined(with: .move(edge: .bottom)))
    }

    private func loadModesForReprocess() {
        reprocessModeList = []
        JunoBroker.getJSON(path: "api/broker/modes/builtin") { builtinObj in
            let builtins: [(id: String, label: String, isCustom: Bool)] =
                ((builtinObj["modes"] as? [[String: Any]]) ?? []).compactMap { raw in
                    guard let id = raw["id"] as? String, !id.isEmpty else { return nil }
                    return (id: id, label: JunoUserFacingCopy.builtinModeTitle(id: id), isCustom: false)
                }
            JunoBroker.getJSON(path: "api/broker/modes/custom") { customObj in
                let customs: [(id: String, label: String, isCustom: Bool)] =
                    ((customObj["modes"] as? [[String: Any]]) ?? []).compactMap { raw in
                        guard let name = raw["name"] as? String, !name.isEmpty,
                              (raw["enabled"] as? Bool) ?? true else { return nil }
                        return (id: "custom:\(name)", label: name, isCustom: true)
                    }
                reprocessModeList = builtins + customs
                if selectedReprocessMode.isEmpty, let first = builtins.first {
                    selectedReprocessMode = first.id
                }
            }
        }
    }

    private func handleReprocessRequestIfNeeded() {
        guard let request = reprocessRequest,
              request.utteranceId == entry.utteranceId else { return }
        loadModesForReprocess()
        showReprocessSheet = true
        onReprocessRequestHandled()
    }

    private func runReprocess() {
        let modeId = selectedReprocessMode
        guard !modeId.isEmpty else { return }

        let isCustom = modeId.hasPrefix("custom:")
        let modeName = isCustom ? String(modeId.dropFirst("custom:".count)) : modeId
        let modeLabel = reprocessModeList.first(where: { $0.id == modeId })?.label ?? modeName

        reprocessResult = nil
        reprocessError = nil
        reprocessResultModeLabel = modeLabel
        isReprocessing = true

        let payload: [String: Any] = [
            "utterance_id": entry.utteranceId,
            "mode_name": modeName,
            "is_custom": isCustom,
        ]
        JunoBroker.postJSON(path: "api/broker/history/reprocess", payload: payload) { resp in
            isReprocessing = false
            if (resp["ok"] as? Bool) == true,
               let transcript = resp["transcript"] as? String, !transcript.isEmpty {
                withAnimation(.easeOut(duration: 0.2)) {
                    reprocessResult = transcript
                }
            } else {
                reprocessError = (resp["error"] as? String) ?? "Re-run failed"
            }
        }
    }

    // MARK: - Actions

    private func replayAudio() {
        audioError = nil
        JunoBroker.fetchBinary(path: "api/broker/audio/\(entry.utteranceId)/replay") { result in
            switch result {
            case .failure(let error):
                audioError = error.localizedDescription
            case .success(let data):
                do {
                    audioPlayer = try AVAudioPlayer(data: data)
                    audioPlayer?.prepareToPlay()
                    audioPlayer?.play()
                } catch {
                    audioError = error.localizedDescription
                }
            }
        }
    }
}

// Settings moved to `JunoSettingsView.swift`

// MARK: - Root shell view

struct JunoMainShellView: View {
    @ObservedObject var surface: SurfaceEditingModel
    @ObservedObject var controller: DictationController
    @ObservedObject private var windowNav = JunoMainWindowNavigator.shared
    @Environment(\.colorScheme) private var scheme
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var homeGreetingStore = JunoHomeGreetingStore()
    @StateObject private var setup = JunoSetupModel()
    @State private var healthRefreshTick: Int = 0
    @State private var homeVisitToken: Int = 1
    @State private var lastSectionForHomeVisit: MainSidebar?
    @State private var isNavigating: Bool = false
    @State private var hoveredSection: MainSidebar?
    @ObservedObject private var updater = JunoUpdater.shared

    var body: some View {
        ZStack {
            shellBackdrop

            HStack(alignment: .top, spacing: 18) {
                premiumSidebar

                NavigationStack {
                    ZStack {
                        detailView
                            .id(windowNav.section)
                            .transition(.opacity)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .animation(.easeOut(duration: 0.18), value: windowNav.section)
                }
                .toolbar(.hidden, for: .automatic)
                .junoBrandWindow()
                .junoStageSurface()
            }
            .padding(18)
            .padding(.top, 10)
            .padding(.bottom, 14)

        }
        // Keep the global Voice Action toast above every page without
        // installing a full-window hit-test layer. Only the toast card
        // itself should receive clicks; empty overlay space must pass
        // through to the app chrome. (Same intent as origin/main's
        // 365bf39 toast fix; the .frame(.infinity) is applied below
        // after the overlay so both effects compose.)
        .overlay(alignment: .bottomTrailing) {
            JunoActionToastOverlay()
                .padding(.trailing, 22)
                .padding(.bottom, 22)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .groupBoxStyle(JunoBrandGroupBoxStyle())
        .junoBrandWindow()
        .toolbar(.hidden, for: .windowToolbar)
        .onAppear {
            if scenePhase == .active || NSApp.isActive {
                resumeForegroundPolling()
            }
        }
        .onDisappear {
            setup.stopPolling()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            resumeForegroundPolling()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.willResignActiveNotification)) { _ in
            setup.stopPolling()
        }
        .onChange(of: scenePhase) { newPhase in
            if newPhase == .active {
                resumeForegroundPolling()
            } else {
                setup.stopPolling()
            }
        }
        .onChange(of: windowNav.section) { newSection in
            isNavigating = true
            // Debounce matches the new (shorter) opacity transition.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.20) {
                isNavigating = false
            }
            if lastSectionForHomeVisit == nil {
                lastSectionForHomeVisit = newSection
                return
            }
            if let prev = lastSectionForHomeVisit, newSection == .home, prev != .home {
                homeVisitToken += 1
            }
            lastSectionForHomeVisit = newSection
        }
    }

    private func resumeForegroundPolling() {
        setup.startPolling()
        surface.refresh()
        healthRefreshTick += 1
    }

    private var shellBackdrop: some View {
        ZStack {
            LinearGradient(
                colors: [
                    JunoTheme.windowBackground(scheme),
                    JunoTheme.windowBackground(scheme).opacity(0.98),
                    Color.black.opacity(scheme == .dark ? 0.22 : 0.02)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
        }
    }

    // MARK: - Premium sidebar

    private var premiumSidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                JunoChromeAmbientMark(large: false, presentation: .editorial, idleBreathing: false)
                Text("Juno")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .padding(.top, 14)
            .padding(.bottom, 12)

            VStack(spacing: 2) {
                ForEach(MainSidebar.allCases) { item in
                    sidebarRow(item)
                }
            }
            .padding(.horizontal, 6)

            Spacer(minLength: 0)

            sidebarFooter
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 12)
        .frame(width: JunoTheme.SplitColumns.mainSidebarIdeal + 12)
        .frame(maxHeight: .infinity)
        .junoSidebarRail()
    }

    private var sidebarFooter: some View {
        VStack(alignment: .leading, spacing: 8) {
            Rectangle()
                .fill(JunoTheme.divider(scheme))
                .frame(height: 1)
                .opacity(0.55)
                .padding(.top, 4)

            if updater.updateAvailable {
                HStack(spacing: 8) {
                    Image(systemName: "arrow.down.circle.fill")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(JunoDesignTokens.accent)
                    Text("Update available")
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Spacer(minLength: 0)
                    Button("Settings") { windowNav.section = .settings }
                    .buttonStyle(.plain)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(JunoDesignTokens.accent)
                }
            }

            Text(JunoProductIdentity.versionSummary)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.88))
                .lineLimit(2)
        }
        .padding(.horizontal, 14)
        .padding(.bottom, 2)
    }

    private func sidebarRow(_ item: MainSidebar) -> some View {
        let selected = windowNav.section == item
        let hovered  = hoveredSection == item

        return Button {
            windowNav.section = item
        } label: {
            HStack(spacing: 10) {
                Image(systemName: item.symbol)
                    .font(.system(size: 13, weight: selected ? .semibold : .regular))
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 18, alignment: .center)
                Text(item.title)
                    .font(.system(size: 12, weight: selected ? .semibold : .regular, design: .rounded))
                    .lineLimit(1)
                Spacer(minLength: 0)
            }
            .foregroundStyle(
                selected
                    ? JunoTheme.primaryText(scheme)
                    : (hovered ? JunoTheme.primaryText(scheme) : JunoTheme.secondaryText(scheme).opacity(scheme == .dark ? 0.88 : 1))
            )
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(
                        selected
                            ? JunoDesignTokens.accent.opacity(scheme == .dark ? 0.18 : 0.12)
                            : (hovered ? Color.white.opacity(scheme == .dark ? 0.07 : 0.05) : Color.clear)
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(
                        selected && scheme == .dark ? Color.white.opacity(0.08) : Color.clear,
                        lineWidth: 0.6
                    )
            )
        }
        .buttonStyle(.plain)
        .focusable(false)
        .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                hoveredSection = hovering ? item : nil
            }
        }
    }

    // MARK: - Detail view

    @ViewBuilder
    private var detailView: some View {
        switch windowNav.section {
        case .home: JunoHomeView(
                surface: surface,
                controller: controller,
                greeting: homeGreetingStore,
                setup: setup,
                homeVisitToken: homeVisitToken,
                isNavigating: isNavigating,
                healthRefreshTick: healthRefreshTick
            ).junoCenteredReadingPane()
        case .actions:         JunoActionsPage().junoCenteredReadingPane()
        case .voiceCommands:   JunoVoiceCommandsPage().junoCenteredReadingPane()
        case .history:         JunoHistorySplitView()
        case .modes:           JunoModesView()
        case .personalization: MemoryManagementView()
        case .surfacePresets:  SurfacePresetsView()
        case .privacy:         JunoPrivacyView().junoCenteredReadingPane()
        case .settings:        JunoSettingsView(setup: setup).junoCenteredReadingPane()
        }
    }
}

// MARK: - Window host

private final class JunoMainWindowCloseDelegate: NSObject, NSWindowDelegate {
    private let onClose: () -> Void
    private let onScreenChanged: (NSWindow) -> Void
    init(onClose: @escaping () -> Void, onScreenChanged: @escaping (NSWindow) -> Void) {
        self.onClose = onClose
        self.onScreenChanged = onScreenChanged
    }
    func windowWillClose(_ notification: Notification) {
        onClose()
        // Restore .accessory activation policy when the main window closes
        // if the user has Show-in-Dock disabled. Window activation may have
        // promoted us to .regular; on close we should fall back to menu-bar-
        // only so the Dock tile doesn't linger after the window is gone.
        if !JunoUserDefaults.showInDock {
            NSApp.setActivationPolicy(.accessory)
        }
    }
    func windowDidChangeScreen(_ notification: Notification) {
        guard let w = notification.object as? NSWindow else { return }
        onScreenChanged(w)
    }

    func windowDidResize(_ notification: Notification) {
        guard let w = notification.object as? NSWindow else { return }
        onScreenChanged(w)
    }
}

enum JunoMainWindow {
    private static var windowController: NSWindowController?
    private static var closeDelegate: JunoMainWindowCloseDelegate?
    // 1080px wide — gives History's detail pane ~510px after sidebar +
    // list column, so detail buttons (Copy / Teach Juno / View full) stop
    // truncating and body text wraps at a comfortable reading measure.
    // Saved frames above 1080 are unaffected; smaller frames grow on next
    // launch.
    private static let preferredContentSize = NSSize(width: 1080, height: 680)
    /// Minimum width for nested splits: main sidebar + primary list column (History, Modes) + detail.
    /// History and Modes use the same two-column inner split; Dictionary adds a category rail inside the detail.
    ///
    /// Detail-pane floor bumped 300 → 430 in the History redesign so the
    /// action-hero card has room to breathe (body wraps to readable
    /// widths, "Open in Notes" doesn't truncate, app name in the header
    /// doesn't shrink to "Goog…"). Net minimum content width: 920.
    /// Saved frames at or above 920 are unaffected; smaller windows get
    /// grown to 920 on next launch.
    private static var layoutMinimumContentWidth: CGFloat {
        JunoTheme.SplitColumns.mainSidebarMin
            + JunoTheme.SplitColumns.primaryListMin
            + 430
    }

    private static let layoutMinimumContentHeight: CGFloat = 480

    @MainActor
    private static func applyScreenAwareCaps(for window: NSWindow) {
        let visible = (window.screen ?? NSScreen.main)?.visibleFrame
        guard let visible else { return }
        // Keep tab content from dictating the NSWindow size. Split-heavy tabs
        // such as History are allowed to compress within the product
        // window; they should not grow the window toward full-screen.
        let capW = max(400, visible.width - 80)
        let capH = max(360, visible.height - 120)
        // Let users resize up to the visible screen bounds; avoid a fixed 940x660 feel.
        let maxW = capW
        let maxH = capH
        let minW = min(layoutMinimumContentWidth, maxW)
        let minH = min(layoutMinimumContentHeight, maxH)
        window.contentMinSize = NSSize(width: minW, height: minH)
        window.contentMaxSize = NSSize(width: maxW, height: maxH)
        clampWindowFrameToVisibleScreen(window)
    }

    @MainActor
    private static func clampWindowFrameToVisibleScreen(_ window: NSWindow) {
        let visible = (window.screen ?? NSScreen.main)?.visibleFrame ?? .zero
        guard visible.width > 1, visible.height > 1 else { return }

        let cMin = window.contentMinSize
        let cMax = window.contentMaxSize

        var frame = window.frame
        if frame.width > cMax.width { frame.size.width = cMax.width }
        if frame.height > cMax.height { frame.size.height = cMax.height }

        // Keep the window fully inside the visible screen rect (best-effort).
        if frame.maxX > visible.maxX { frame.origin.x = visible.maxX - frame.width }
        if frame.maxY > visible.maxY { frame.origin.y = visible.maxY - frame.height }
        if frame.minX < visible.minX { frame.origin.x = visible.minX }
        if frame.minY < visible.minY { frame.origin.y = visible.minY }

        if frame.width > visible.width {
            frame.origin.x = visible.minX
        }
        if frame.height > visible.height {
            frame.origin.y = visible.minY
        }

        // Never leave the frame narrower than contentMinSize (avoids clipped NavigationSplitView columns).
        frame.size.width = max(cMin.width, min(frame.size.width, cMax.width, visible.width))
        frame.size.height = max(cMin.height, min(frame.size.height, cMax.height, visible.height))

        if frame.maxX > visible.maxX { frame.origin.x = visible.maxX - frame.width }
        if frame.maxY > visible.maxY { frame.origin.y = visible.maxY - frame.height }
        if frame.minX < visible.minX { frame.origin.x = visible.minX }
        if frame.minY < visible.minY { frame.origin.y = visible.minY }

        if frame != window.frame {
            window.setFrame(frame, display: true, animate: false)
        }
    }

    @MainActor
    static func activateIfPresent() -> Bool {
        guard let wc = windowController, let w = wc.window else { return false }
        bringToFront(w)
        return true
    }

    @MainActor
    private static func bringToFront(_ window: NSWindow) {
        JunoWindowActivation.bringToFront(window)
    }

    @MainActor
    static func show(surface: SurfaceEditingModel, controller: DictationController, section: MainSidebar = .home) {
        guard JunoUserDefaults.onboardingCompleted else {
            JunoOnboardingWindow.showIfNeeded()
            JunoWindowActivation.activateApp()
            return
        }
        JunoMainWindowNavigator.shared.section = section
        if let wc = windowController, let w = wc.window, w.isVisible {
            bringToFront(w)
            return
        }
        windowController = nil; closeDelegate = nil
        let root = JunoMainShellView(surface: surface, controller: controller)
        let hosting = NSHostingController(rootView: root)
        let window = NSWindow(contentViewController: hosting)
        window.title = ""
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.setContentSize(preferredContentSize)
        window.contentMinSize = NSSize(width: layoutMinimumContentWidth, height: layoutMinimumContentHeight)
        window.contentMaxSize = preferredContentSize
        applyScreenAwareCaps(for: window)
        window.setFrameAutosaveName("JunoMainWindow")
        if !window.setFrameUsingName("JunoMainWindow") { window.center() }
        applyScreenAwareCaps(for: window)
        window.isReleasedWhenClosed = false
        let del = JunoMainWindowCloseDelegate {
            JunoMainWindow.windowController = nil
            JunoMainWindow.closeDelegate = nil
        } onScreenChanged: { w in
            JunoMainWindow.applyScreenAwareCaps(for: w)
        }
        closeDelegate = del; window.delegate = del
        let wc = NSWindowController(window: window)
        windowController = wc
        wc.showWindow(nil)
        // SwiftUI may attach a toolbar for navigation chrome; keep the titlebar visually blank.
        window.toolbar = nil
        applyScreenAwareCaps(for: window)
        bringToFront(window)
    }
}

// MARK: - UtteranceHistoryEntry display helpers

extension UtteranceHistoryEntry {
    var historyTimestampLabel: String {
        guard let ts = tsUnixMs else { return "Now" }
        let date = Date(timeIntervalSince1970: Double(ts) / 1000.0)
        let fmt = DateFormatter()
        fmt.locale = Locale.current
        if Calendar.current.isDateInToday(date) {
            fmt.dateFormat = "HH:mm"
        } else if Calendar.current.isDateInYesterday(date) {
            fmt.dateFormat = "EEE HH:mm"
        } else {
            fmt.dateFormat = "MMM d"
        }
        return fmt.string(from: date)
    }

    var displayTimestamp: String? {
        guard let ts = tsUnixMs else { return nil }
        let date = Date(timeIntervalSince1970: Double(ts) / 1000.0)
        let cal = Calendar.current
        let fmt = DateFormatter()
        if cal.isDateInToday(date)     { fmt.dateFormat = "HH:mm" }
        else if cal.isDateInYesterday(date) { fmt.dateFormat = "'Yesterday' HH:mm" }
        else { fmt.dateStyle = .short; fmt.timeStyle = .short }
        return fmt.string(from: date)
    }

    var historyHeaderTitle: String {
        isActionHistoryRow ? "Voice Action" : displayAppName
    }

    var historyHeaderSubtitle: String? {
        if isActionHistoryRow {
            var parts: [String] = []
            let app = displayAppName.trimmingCharacters(in: .whitespacesAndNewlines)
            if !app.isEmpty && app != "Unknown" {
                parts.append("From \(app)")
            }
            if let ts = displayTimestamp {
                parts.append(ts)
            }
            return parts.isEmpty ? nil : parts.joined(separator: " · ")
        }
        return displayTimestamp
    }

    var wordCountLabel: String? {
        guard let t = transcript else { return nil }
        let n = t.split { $0.isWhitespace }.filter { !$0.isEmpty }.count
        return n > 0 ? "\(n) words" : nil
    }

    var displayFailureReason: String? {
        guard let reason = normalizedFailureReason else { return nil }
        if reason.hasPrefix("capability_blocked") {
            return "Juno is blocked from inserting text in \(displayAppName)."
        }

        switch reason {
        case "paste_failed":
            return "Juno could not insert text in \(displayAppName)."
        case "undo_safe_paste_failed":
            return "Juno could not safely paste into \(displayAppName)."
        case "no_active_text_field":
            return "No editable text field was active in \(displayAppName)."
        case "paste_kind_none_with_text":
            return "Juno captured text, but insertion was disabled for this target."
        case "ax_permission_missing":
            return "Accessibility permission is missing, so Juno could not insert text."
        case "empty_audio":
            return "Juno did not hear enough audio to transcribe."
        case "user_cancelled_hud":
            return "Dictation was cancelled before insertion."
        case "broker_unreachable":
            return "The local voice engine was unreachable."
        default:
            return Self.humanizedFailureCode(reason)
        }
    }

    private var normalizedFailureReason: String? {
        let trimmed = failureReason?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

    private var hasTranscriptText: Bool {
        transcript?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
    }

    private static func humanizedFailureCode(_ code: String) -> String {
        code
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: ":", with: ": ")
            .capitalized
    }

    var historyPreviewText: String {
        let base = hasTranscriptText
            ? transcript
            : displayFailureReason
        let trimmed = base?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? "No transcript available" : trimmed
    }

    var isActionHistoryRow: Bool {
        !(actions ?? []).isEmpty
    }

    var historyPrimaryLine: String {
        actionHistorySummary ?? historyPreviewText
    }

    var actionHistorySummary: String? {
        let actions = actionResultsForHistoryDisplay(preserveFreshPending: true)
        guard !actions.isEmpty else { return nil }
        if actions.count == 1, let action = actions.first {
            let body = action.bodyPreview.trimmingCharacters(in: .whitespacesAndNewlines)
            return body.isEmpty
                ? action.kind.descriptor.displayName
                : "\(action.kind.descriptor.displayName): \(body)"
        }
        let firstBody = actions.first?.bodyPreview.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return firstBody.isEmpty ? "\(actions.count) voice actions" : firstBody
    }

    var historySecondaryLine: String {
        var parts: [String] = []
        if isActionHistoryRow {
            parts.append(actionHistoryOutcomeLine ?? "Voice Action")
        }
        if let wc = wordCountLabel { parts.append(wc) }
        if let ms = processingMs, ms > 0 {
            parts.append(String(format: "%.1fs", Double(ms) / 1000.0))
        }
        if let mode, !mode.isEmpty {
            parts.append(JunoUserFacingCopy.builtinModeTitle(id: mode))
        }
        if !isActionHistoryRow, hasTranscriptText, let reason = displayFailureReason, !reason.isEmpty {
            parts.append(reason)
        }
        return parts.isEmpty ? "Saved on this Mac" : parts.joined(separator: "  •  ")
    }

    var actionHistoryOutcomeLine: String? {
        let actions = actionResultsForHistoryDisplay(preserveFreshPending: true)
        guard !actions.isEmpty else { return nil }
        let summary = JunoActionBatchFormatter.summarize(actions)
        switch summary.tone {
        case .allSaved, .allPending:
            return summary.headline
        case .blocked, .failed, .partial:
            return summary.oneLine
        }
    }

    func actionResultsForHistoryDisplay(
        activeUtteranceId: String? = nil,
        preserveFreshPending: Bool = false
    ) -> [JunoActionResult] {
        let raw = actions ?? []
        guard !raw.isEmpty else { return [] }
        let active = activeUtteranceId == utteranceId
        let isFresh = isFreshPendingActionRow
        return raw.map { action in
            guard action.status == .pending else { return action }
            if active { return action }
            if preserveFreshPending && isFresh && action.hasExplicitStatus {
                return action
            }
            return action.withDisplayStatus(
                .sinkError,
                error: "This action was parsed but was not saved. Retry it from History."
            )
        }
    }

    private var isFreshPendingActionRow: Bool {
        guard let ts = updatedAtMs ?? tsUnixMs else { return false }
        let ageMs = Int64(Date().timeIntervalSince1970 * 1000.0) - ts
        return ageMs >= 0 && ageMs < 45_000
    }

    var showsRewriteSection: Bool {
        let raw = rawTranscript?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let fin = transcript?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !raw.isEmpty, !fin.isEmpty else { return false }
        return raw != fin
    }

    /// Up to 6 likely-vocabulary candidates extracted from the transcript:
    /// proper nouns (capitalized words that aren't sentence starters),
    /// ALL_CAPS acronyms (BOFA, QBR), and CamelCase tokens. Used to
    /// pre-populate the Save Phrase chip strip — much more useful than
    /// dumping the whole transcript into a "Term" field.
    var vocabularyCandidates: [String] {
        let source = transcript?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !source.isEmpty else { return [] }

        // Tokenize on whitespace + punctuation so we don't carry trailing
        // commas/periods into candidates.
        let separators = CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters)
        let tokens = source.components(separatedBy: separators).filter { !$0.isEmpty }

        var seen = Set<String>()
        var candidates: [String] = []
        let stopwords: Set<String> = ["A", "An", "And", "As", "At", "But", "For", "From", "I", "If", "In", "Is", "It", "My", "Of", "On", "Or", "So", "That", "The", "This", "To", "We", "You"]

        for (idx, raw) in tokens.enumerated() {
            // Skip pure punctuation / single chars.
            guard raw.count >= 2 else { continue }

            // ALL_CAPS acronyms (3+ letters): always include.
            if raw == raw.uppercased(), raw.rangeOfCharacter(from: .letters) != nil, raw.count >= 3 {
                if seen.insert(raw).inserted { candidates.append(raw) }
                continue
            }

            // Proper nouns: starts uppercase, has lowercase too. Skip if
            // it's the very first word of the utterance (sentence start),
            // and skip common sentence-starter words even mid-sentence.
            let first = raw.first!
            let isCapitalized = first.isUppercase && raw.dropFirst().contains(where: { $0.isLowercase })
            guard isCapitalized else { continue }
            if idx == 0 { continue }
            if stopwords.contains(raw) { continue }
            if seen.insert(raw).inserted { candidates.append(raw) }
            if candidates.count >= 6 { break }
        }
        return candidates
    }

    var outcomeColor: Color {
        if let actions, !actions.isEmpty {
            switch JunoActionBatchFormatter.summarize(actions).tone {
            case .allSaved: return JunoDesignTokens.meadow
            case .allPending, .partial: return JunoDesignTokens.accent
            case .blocked, .failed: return JunoDesignTokens.danger
            }
        }
        if normalizedFailureReason != nil { return JunoDesignTokens.danger }
        if (transcript?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false) {
            return JunoDesignTokens.meadow
        }
        return JunoDesignTokens.muted
    }

    var transcriptBodyHeightCap: CGFloat {
        let source = transcript?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !source.isEmpty else { return 120 }
        let explicitBreaks = source.reduce(into: 0) { count, char in
            if char == "\n" { count += 1 }
        }
        let estimatedLines = max(4, explicitBreaks + Int(ceil(Double(source.count) / 62.0)))
        let estimatedHeight = CGFloat(estimatedLines) * 18.0
        return min(max(estimatedHeight, 120), 320)
    }
}
