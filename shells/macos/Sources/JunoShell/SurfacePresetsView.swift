import AppKit
import SwiftUI

/// Built-in mode ids when the broker list is unavailable.
private let kFallbackBuiltinModeRows: [(String, String)] = [
    "verbatim", "casual_chat", "formal_email", "structured_notes", "explicit_rewrite", "command_mode",
].map { id in (id, JunoUserFacingCopy.builtinModeTitle(id: id)) }

/// One-liner that explains a built-in writing style — used inside the
/// custom Writing-style dropdown so the user picks with intent, not by guess.
private func writingStyleSummary(for id: String) -> String {
    switch id.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "verbatim":         return "Keep words exactly as said."
    case "casual_chat":      return "Light cleanup, conversational tone."
    case "formal_email":     return "Polished, complete sentences."
    case "structured_notes": return "Headings and bullets, skimmable."
    case "explicit_rewrite": return "Stronger rewrite for flow and clarity."
    case "command_mode":     return "Voice commands, not dictation."
    default:                 return "Built-in writing style."
    }
}

/// Snapshot of the user's global privacy choices, used to compute the
/// inheritance hint for each per-app row ("Following your global setting").
private struct GlobalPrivacy: Equatable {
    var useContext: Bool = true
    var learn: Bool = true
    var saveHistory: Bool = true
    var saveAudio: Bool = true
    /// True once the broker calls have returned at least once. Until then,
    /// inheritance hints stay neutral so the UI doesn't flicker through a
    /// wrong default.
    var loaded: Bool = false
}

/// Per-app defaults for how Juno writes. Auto-saves on every change — no Save
/// button. The page is anchored to a chosen app: when nothing is picked, the
/// settings sections stay hidden so the user never sees a stray "Formal email"
/// or "Use global" row that looks active without context.
struct SurfacePresetsView: View {
    @ObservedObject private var windowNav = JunoMainWindowNavigator.shared
    @ObservedObject private var lifecycle = JunoEngineLifecycle.shared

    // Data
    @State private var userRows: [[String: Any]] = []
    @State private var brokerReachable: Bool = true
    @State private var loadError: String?
    @State private var operationError: (title: String, message: String)?
    @State private var runningApps: [(bundle: String, name: String, icon: NSImage?)] = []
    @State private var builtinModes: [(String, String)] = []
    @State private var activeNowLabel: String?
    @State private var globalPrivacy = GlobalPrivacy()

    // Form
    @State private var selectedBundle: String = ""
    @State private var selectedModeId: String = ""
    @State private var advancedOpen = false
    @State private var advancedRuleId = ""
    @State private var asrField = ""
    @State private var toneField = ""
    @State private var includeTitle = false
    /// Per-app override states. Each is one of "default" | "on" | "off".
    @State private var appUseContext = "default"
    @State private var appLearn = "default"
    @State private var appSaveHistory = "default"
    @State private var appSaveAudio = "default"

    // Async + race guard
    @State private var hydratedBundle: String = ""
    @State private var savingPreset = false
    @State private var savingPrivacy = false

    // Custom-app sheet (only when user explicitly opts in)
    @State private var showAddAppSheet = false
    @State private var pendingCustomBundle = ""

    @Environment(\.colorScheme) private var scheme

    private var hasApp: Bool { !selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    private var selectedAppDisplay: (name: String, icon: NSImage?)? {
        let b = selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !b.isEmpty else { return nil }
        if let row = runningApps.first(where: { $0.bundle == b }) {
            return (row.name, row.icon)
        }
        if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: b) {
            let name = url.deletingPathExtension().lastPathComponent
            let icon = NSWorkspace.shared.icon(forFile: url.path)
            return (name, icon)
        }
        return (b, nil)
    }

    private var presetIdForCurrentBundle: String {
        let manual = advancedRuleId.trimmingCharacters(in: .whitespacesAndNewlines)
        if !manual.isEmpty { return manual }
        let b = selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !b.isEmpty else { return "" }
        return "user_" + b.replacingOccurrences(of: ".", with: "_").lowercased()
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: JunoTheme.Density.cardGap) {
                JunoPageHeader(
                    eyebrow: "Per-app writing",
                    title: "Choose how Juno writes in each app",
                    subtitle: "Pick an app, then set the style and privacy you want when you’re using it.",
                    trailing: {
                        Button {
                            refreshRunningApps()
                            reload()
                            loadBuiltinModes()
                            loadGlobalPrivacy()
                        } label: {
                            Label("Refresh", systemImage: "arrow.clockwise")
                        }
                        .junoSecondaryActionButton()
                    }
                )

                banners

                appPickerCard

                if hasApp {
                    writingStyleCard
                    privacyCard
                    advancedCard
                    if hasExistingRule { deleteFooter }
                } else {
                    emptyStateCard
                }

                if !userRows.isEmpty {
                    yourAppsCard
                }
            }
            .junoDetailPagePadding()
            .frame(maxWidth: 720)
        }
        .sheet(isPresented: $showAddAppSheet) { addAppSheet }
        .onAppear {
            JunoBroker.pingHealth { ok in brokerReachable = ok }
            refreshRunningApps()
            loadBuiltinModes()
            loadGlobalPrivacy()
            reload()
            applyPendingBundleIfNeeded()
            refreshActiveSurface()
        }
        .onReceive(lifecycle.$phase) { _ in
            JunoBroker.pingHealth { ok in brokerReachable = ok }
        }
        .onChange(of: windowNav.pendingPresetBundleId) { newVal in
            guard let raw = newVal?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return }
            refreshRunningApps()
            selectBundleForEditing(raw)
            windowNav.pendingPresetBundleId = nil
        }
    }

    // MARK: - Banners

    @ViewBuilder
    private var banners: some View {
        if !brokerReachable {
            JunoInlineStatusBanner(
                kind: .warning,
                title: "Voice engine not connected",
                message: "Per-app choices can’t load or save until Juno reconnects.",
                systemImage: "wifi.slash"
            )
        }
        if let active = activeNowLabel, !active.isEmpty {
            JunoInlineStatusBanner(
                kind: .success,
                title: active,
                message: nil,
                systemImage: "checkmark.circle.fill"
            )
        }
        if let err = loadError, !err.isEmpty {
            JunoInlineStatusBanner(
                kind: .danger,
                title: "Couldn’t load your app choices",
                message: err,
                systemImage: "exclamationmark.triangle.fill"
            ) {
                Button("Dismiss") { loadError = nil }.junoTertiaryActionButton()
            }
        }
        if let opErr = operationError {
            JunoInlineStatusBanner(
                kind: .danger,
                title: opErr.title,
                message: opErr.message,
                systemImage: "exclamationmark.triangle.fill"
            ) {
                Button("Dismiss") { operationError = nil }.junoTertiaryActionButton()
            }
        }
    }

    // MARK: - App picker card

    private var appPickerCard: some View {
        VStack(alignment: .leading, spacing: JunoUI.Spacing.s) {
            JunoSectionLabel(text: "App")
            AppPickerButton(
                selection: $selectedBundle,
                runningApps: runningApps,
                resolvedDisplay: selectedAppDisplay,
                onPick: { bundle in selectBundleForEditing(bundle) },
                onAddCustom: {
                    pendingCustomBundle = ""
                    showAddAppSheet = true
                }
            )
        }
        .padding(JunoTheme.Density.cardPadding)
        .premiumCard()
    }

    // MARK: - Writing style

    private var writingStyleCard: some View {
        VStack(alignment: .leading, spacing: JunoUI.Spacing.s) {
            JunoSectionLabel(text: "How Juno writes here")
            Text("The default style Juno uses when you’re in this app.")
                .junoType(.caption)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
            WritingStyleButton(
                selectedId: selectedModeId,
                modes: builtinModes,
                onPick: { id in
                    selectedModeId = id
                    savePresetIfReady()
                }
            )
            if savingPreset { inlineSavingHint }
        }
        .padding(JunoTheme.Density.cardPadding)
        .premiumCard()
    }

    // MARK: - Privacy

    private var privacyCard: some View {
        VStack(alignment: .leading, spacing: JunoUI.Spacing.m) {
            JunoSectionLabel(text: "Privacy here")
            Text("Override your global privacy choices when you’re in this app.")
                .junoType(.caption)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
            VStack(spacing: JunoUI.Spacing.s) {
                privacyRow(
                    title: "Use context",
                    subtitle: "Lets Juno read selected text and clipboard hints.",
                    override: $appUseContext,
                    globalValue: globalPrivacy.useContext
                )
                Divider().opacity(0.25)
                privacyRow(
                    title: "Learn corrections",
                    subtitle: "Save your edits so future writing improves.",
                    override: $appLearn,
                    globalValue: globalPrivacy.learn
                )
                Divider().opacity(0.25)
                privacyRow(
                    title: "Save history",
                    subtitle: "Keep a local record of what you dictated here.",
                    override: $appSaveHistory,
                    globalValue: globalPrivacy.saveHistory
                )
                Divider().opacity(0.25)
                privacyRow(
                    title: "Keep recordings",
                    subtitle: "Hold audio for replay and troubleshooting.",
                    override: $appSaveAudio,
                    globalValue: globalPrivacy.saveAudio
                )
            }
            if savingPrivacy { inlineSavingHint }
        }
        .padding(JunoTheme.Density.cardPadding)
        .premiumCard()
    }

    private func privacyRow(
        title: String,
        subtitle: String,
        override: Binding<String>,
        globalValue: Bool
    ) -> some View {
        PerAppPrivacyRow(
            title: title,
            subtitle: subtitle,
            override: override.wrappedValue,
            globalValue: globalValue,
            globalLoaded: globalPrivacy.loaded,
            onToggle: { newOn in
                override.wrappedValue = newOn ? "on" : "off"
                saveAppPrivacyIfReady()
            },
            onReset: {
                override.wrappedValue = "default"
                saveAppPrivacyIfReady()
            }
        )
    }

    // MARK: - Advanced

    private var advancedCard: some View {
        DisclosureGroup(isExpanded: $advancedOpen) {
            VStack(alignment: .leading, spacing: JunoUI.Spacing.m) {
                Toggle("Include window title when listening", isOn: $includeTitle)
                    .onChange(of: includeTitle) { _ in savePresetIfReady() }
                VStack(alignment: .leading, spacing: JunoUI.Spacing.xs) {
                    Text("Listening hint")
                        .junoType(.label)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    bordered(TextEditor(text: $asrField).focusEffectDisabled().frame(minHeight: 56))
                }
                VStack(alignment: .leading, spacing: JunoUI.Spacing.xs) {
                    Text("Tone hint")
                        .junoType(.label)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    bordered(TextEditor(text: $toneField).focusEffectDisabled().frame(minHeight: 48))
                }
                HStack {
                    Spacer()
                    Button("Save hints") { savePresetIfReady() }
                        .junoSecondaryActionButton()
                        .disabled(savingPreset || !canSavePreset)
                }
            }
            .padding(.top, JunoUI.Spacing.s)
        } label: {
            Text("Advanced")
                .junoType(.bodyEmphasis)
                .foregroundStyle(JunoTheme.primaryText(scheme))
        }
        .padding(JunoTheme.Density.cardPadding)
        .premiumCard()
    }

    // MARK: - Empty state

    private var emptyStateCard: some View {
        VStack(spacing: JunoUI.Spacing.s) {
            Image(systemName: "app.badge")
                .font(.system(size: 28, weight: .regular))
                .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.7))
            Text("Pick an app to get started")
                .junoType(.bodyEmphasis)
                .foregroundStyle(JunoTheme.primaryText(scheme))
            Text("Choose an app above and Juno will use that style and privacy whenever you’re in it.")
                .junoType(.caption)
                .multilineTextAlignment(.center)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .frame(maxWidth: 360)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, JunoUI.Spacing.xl)
        .padding(.horizontal, JunoTheme.Density.cardPadding)
        .premiumCard()
    }

    // MARK: - Delete footer (only when an actual rule exists for selected app)

    private var hasExistingRule: Bool {
        let target = selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !target.isEmpty else { return false }
        return userRows.contains(where: { strValue($0["bundle_id"]).lowercased() == target })
    }

    private var deleteFooter: some View {
        HStack {
            Spacer()
            Button(role: .destructive) {
                deletePresetForCurrentApp()
            } label: {
                Label("Remove this app’s rule", systemImage: "trash")
            }
            .junoTertiaryActionButton()
        }
    }

    // MARK: - "Your apps" list (existing rules)

    private var yourAppsCard: some View {
        VStack(alignment: .leading, spacing: JunoUI.Spacing.s) {
            JunoSectionLabel(text: "Your apps")
            VStack(spacing: 0) {
                ForEach(Array(userRows.enumerated()), id: \.offset) { idx, row in
                    appRuleRow(row)
                    if idx < userRows.count - 1 {
                        Divider().opacity(0.25)
                    }
                }
            }
        }
        .padding(JunoTheme.Density.cardPadding)
        .premiumCard()
    }

    private func appRuleRow(_ row: [String: Any]) -> some View {
        let bundle = strValue(row["bundle_id"])
        let mode = strValue(row["default_built_in_mode"])
        let modeTitle = JunoUserFacingCopy.builtinModeTitle(id: mode)
        let display: (name: String, icon: NSImage?) = {
            if let r = runningApps.first(where: { $0.bundle == bundle }) { return (r.name, r.icon) }
            if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundle) {
                return (url.deletingPathExtension().lastPathComponent, NSWorkspace.shared.icon(forFile: url.path))
            }
            return (bundle, nil)
        }()
        let isCurrent = bundle.lowercased() == selectedBundle.lowercased() && !bundle.isEmpty
        return Button {
            selectBundleForEditing(bundle)
        } label: {
            HStack(spacing: JunoUI.Spacing.m) {
                appIconView(display.icon, size: 28)
                VStack(alignment: .leading, spacing: 2) {
                    Text(display.name)
                        .junoType(.bodyEmphasis)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text(modeTitle)
                        .junoType(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
                Spacer()
                if isCurrent {
                    Text("Editing")
                        .junoType(.caption)
                        .foregroundStyle(JunoDesignTokens.accent)
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(Capsule().fill(JunoDesignTokens.accent.opacity(0.10)))
                } else {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.6))
                }
            }
            .padding(.vertical, JunoUI.Spacing.s)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Add custom app sheet

    private var addAppSheet: some View {
        VStack(alignment: .leading, spacing: JunoUI.Spacing.m) {
            VStack(alignment: .leading, spacing: JunoUI.Spacing.xs) {
                Text("Add an app that isn’t open")
                    .junoType(.subtitle)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("Type the app’s identifier — usually something like com.apple.mail. Or just open the app and pick it from the list above.")
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            TextField("com.apple.mail", text: $pendingCustomBundle)
                .textFieldStyle(.roundedBorder)
                .focusEffectDisabled()
                .frame(maxWidth: .infinity)
            HStack {
                Spacer()
                Button("Cancel") { showAddAppSheet = false }
                    .junoSecondaryActionButton()
                Button("Add") {
                    let trimmed = pendingCustomBundle.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { return }
                    showAddAppSheet = false
                    selectBundleForEditing(trimmed)
                }
                .junoPrimaryActionButton()
                .disabled(pendingCustomBundle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 420)
    }

    // MARK: - Helpers

    private var inlineSavingHint: some View {
        HStack(spacing: 6) {
            ProgressView().controlSize(.small).scaleEffect(0.8)
            Text("Saving…")
                .junoType(.caption)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
        }
    }

    private func bordered<V: View>(_ inner: V) -> some View {
        inner
            .font(.system(.body, design: .rounded))
            .padding(4)
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(JunoTheme.border(scheme).opacity(0.5), lineWidth: 0.5)
            )
    }

    private func appIconView(_ icon: NSImage?, size: CGFloat) -> some View {
        Group {
            if let icon {
                Image(nsImage: icon).resizable().frame(width: size, height: size).cornerRadius(size * 0.22)
            } else {
                RoundedRectangle(cornerRadius: size * 0.22, style: .continuous)
                    .fill(JunoTheme.elevatedCard(scheme))
                    .frame(width: size, height: size)
                    .overlay(
                        Image(systemName: "app")
                            .font(.system(size: size * 0.5))
                            .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.6))
                    )
            }
        }
    }

    private func strValue(_ raw: Any?) -> String {
        (raw as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    // MARK: - Data flow

    private func applyPendingBundleIfNeeded() {
        guard let pending = windowNav.pendingPresetBundleId?.trimmingCharacters(in: .whitespacesAndNewlines),
              !pending.isEmpty
        else { return }
        windowNav.pendingPresetBundleId = nil
        selectBundleForEditing(pending)
    }

    private func refreshRunningApps() {
        var seen = Set<String>()
        var rows: [(String, String, NSImage?)] = []
        for app in NSWorkspace.shared.runningApplications where app.activationPolicy == .regular {
            guard let bid = app.bundleIdentifier, !bid.isEmpty,
                  let name = app.localizedName, !name.isEmpty else { continue }
            if seen.insert(bid).inserted {
                rows.append((bid, name, app.icon))
            }
        }
        rows.sort { $0.1.localizedCaseInsensitiveCompare($1.1) == .orderedAscending }
        runningApps = rows
    }

    private func refreshActiveSurface() {
        JunoBroker.getJSON(path: "api/broker/surface/active") { obj in
            let a = obj["active"] as? [String: Any] ?? [:]
            let app = (a["app_name"] as? String) ?? ""
            let mode = (a["mode"] as? String) ?? ""
            let title = JunoUserFacingCopy.builtinModeTitle(id: mode)
            if !app.isEmpty, !title.isEmpty {
                activeNowLabel = "Active now: \(app) → \(title)"
            } else if !app.isEmpty {
                activeNowLabel = "Active now: \(app)"
            } else {
                activeNowLabel = nil
            }
        }
    }

    private func loadBuiltinModes() {
        JunoBroker.getJSON(path: "api/broker/modes/builtin") { obj in
            let parsed: [(String, String)]
            if (obj["ok"] as? Bool) != false,
               let modes = obj["modes"] as? [[String: Any]] {
                parsed = modes.compactMap { m in
                    guard let id = m["id"] as? String else { return nil }
                    if id == "default_surface" { return nil }
                    return (id, JunoUserFacingCopy.builtinModeTitle(id: id))
                }
            } else {
                parsed = []
            }
            builtinModes = parsed.isEmpty ? kFallbackBuiltinModeRows : parsed
        }
    }

    /// Fetches the user's global privacy + retention so we can show the
    /// "Following your global setting" hint accurately. Two calls (context
    /// settings + retention settings) merge into a single ``GlobalPrivacy``.
    private func loadGlobalPrivacy() {
        JunoBroker.getJSON(path: "api/broker/privacy/context_settings") { obj in
            let settings = (obj["settings"] as? [String: Any]) ?? [:]
            globalPrivacy.useContext = (settings["smart_context"] as? Bool) ?? true
            globalPrivacy.learn = (settings["learn_from_corrections"] as? Bool) ?? true
            globalPrivacy.loaded = true
        }
        JunoBroker.getJSON(path: "api/broker/settings") { obj in
            let settings = (obj["settings"] as? [String: Any]) ?? [:]
            // "off" means the user globally disabled keeping that data;
            // anything else (forever / N days) counts as on.
            let historyPolicy = (settings["history_retention_policy"] as? String) ?? ""
            let audioPolicy = (settings["audio_retention_policy"] as? String) ?? ""
            globalPrivacy.saveHistory = historyPolicy != "off"
            globalPrivacy.saveAudio = audioPolicy != "off"
            globalPrivacy.loaded = true
        }
    }

    private func reload() {
        loadError = nil
        JunoBroker.getJSON(path: "api/broker/surface_presets/user") { obj in
            if (obj["ok"] as? Bool) == false {
                loadError = (obj["error"] as? String) ?? "load_failed"
            }
            userRows = (obj["presets"] as? [[String: Any]]) ?? []
            if hasApp { hydrateForCurrentBundle() }
        }
    }

    private func selectBundleForEditing(_ rawBundle: String) {
        let bundle = rawBundle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !bundle.isEmpty else { return }
        selectedBundle = bundle
        hydrateForCurrentBundle()
    }

    private func hydrateForCurrentBundle() {
        let bundle = selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !bundle.isEmpty else {
            resetFormFields()
            hydratedBundle = ""
            return
        }
        let target = bundle.lowercased()
        let existing = userRows.first(where: { strValue($0["bundle_id"]).lowercased() == target })
        if let row = existing {
            selectedModeId = strValue(row["default_built_in_mode"])
            advancedRuleId = strValue(row["id"])
            asrField = strValue(row["asr_addon"])
            toneField = strValue(row["writer_tone_addon"])
            includeTitle = (row["include_window_title_in_asr"] as? Bool) ?? false
        } else {
            // No saved rule yet — leave the writing-style picker empty so the
            // user picks intentionally rather than seeing a stray default.
            selectedModeId = ""
            advancedRuleId = ""
            asrField = ""
            toneField = ""
            includeTitle = false
        }
        loadPrivacyOverridesForCurrentBundle()
        hydratedBundle = bundle
    }

    private func resetFormFields() {
        selectedModeId = ""
        advancedRuleId = ""
        asrField = ""
        toneField = ""
        includeTitle = false
        appUseContext = "default"
        appLearn = "default"
        appSaveHistory = "default"
        appSaveAudio = "default"
    }

    private func loadPrivacyOverridesForCurrentBundle() {
        let bundle = selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !bundle.isEmpty else { return }
        JunoBroker.getJSON(path: "api/broker/privacy/app_overrides") { obj in
            // Only apply if the user hasn't switched apps mid-flight.
            guard selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines) == bundle else { return }
            let perApp = (obj["per_app"] as? [String: Any]) ?? [:]
            let raw = (perApp[bundle] as? [String: Any]) ?? [:]
            appUseContext = normalizedPrivacyChoice(raw["use_context"])
            appLearn = normalizedPrivacyChoice(raw["learn"])
            appSaveHistory = normalizedPrivacyChoice(raw["save_history"])
            appSaveAudio = normalizedPrivacyChoice(raw["save_audio"])
        }
    }

    private func normalizedPrivacyChoice(_ raw: Any?) -> String {
        if let bool = raw as? Bool { return bool ? "on" : "off" }
        let val = String(describing: raw ?? "default").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["default", "on", "off"].contains(val) ? val : "default"
    }

    // MARK: - Save (auto)

    private var canSavePreset: Bool {
        brokerReachable
            && !selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !selectedModeId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var canSavePrivacy: Bool {
        brokerReachable && !selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func savePresetIfReady() {
        guard canSavePreset else { return }
        savingPreset = true
        operationError = nil
        let bundle = selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines)
        let payload: [String: Any] = [
            "id": presetIdForCurrentBundle,
            "bundle_id": bundle,
            "default_built_in_mode": selectedModeId.trimmingCharacters(in: .whitespacesAndNewlines),
            "asr_addon": asrField,
            "writer_tone_addon": toneField,
            "include_window_title_in_asr": includeTitle,
            "lock_mode": false,
            "enabled": true,
        ]
        JunoBroker.postJSON(path: "api/broker/surface_presets/upsert", payload: payload) { resp in
            savingPreset = false
            if (resp["ok"] as? Bool) == true {
                reload()
            } else {
                operationError = ("Couldn’t save your choice",
                                  (resp["error"] as? String) ?? "We couldn’t save that. Try again in a moment.")
            }
        }
    }

    private func saveAppPrivacyIfReady() {
        guard canSavePrivacy else { return }
        savingPrivacy = true
        operationError = nil
        let bundle = selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines)
        let payload: [String: Any] = [
            "bundle_id": bundle,
            "use_context": appUseContext,
            "learn": appLearn,
            "save_history": appSaveHistory,
            "save_audio": appSaveAudio,
        ]
        JunoBroker.postJSON(path: "api/broker/privacy/app_overrides", payload: payload) { resp in
            savingPrivacy = false
            if (resp["ok"] as? Bool) != true {
                operationError = ("Couldn’t save privacy preference",
                                  (resp["error"] as? String) ?? "We couldn’t save that. Try again in a moment.")
            }
        }
    }

    private func deletePresetForCurrentApp() {
        let target = selectedBundle.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !target.isEmpty else { return }
        guard let row = userRows.first(where: { strValue($0["bundle_id"]).lowercased() == target }) else {
            return
        }
        let pid = strValue(row["id"])
        guard !pid.isEmpty else { return }
        operationError = nil
        JunoBroker.postJSON(path: "api/broker/surface_presets/delete", payload: ["id": pid]) { resp in
            if (resp["ok"] as? Bool) == true {
                reload()
            } else {
                operationError = ("Couldn’t reset this app’s preferences",
                                  (resp["error"] as? String) ?? "We couldn’t save that. Try again in a moment.")
            }
        }
    }
}

// MARK: - Per-app privacy row (toggle + inheritance hint + reset)

/// One privacy setting per app, rendered as a single Toggle that shows the
/// *effective* state for this app (whether inherited from the user's global
/// or set explicitly here). The 3-state data model (default/on/off) is hidden
/// from the UI: the toggle is binary, and an inheritance line below tells the
/// user whether they're inheriting or overriding. "Reset to global" appears
/// only when an override is active.
private struct PerAppPrivacyRow: View {
    let title: String
    let subtitle: String
    let override: String       // "default" | "on" | "off"
    let globalValue: Bool
    let globalLoaded: Bool
    let onToggle: (Bool) -> Void
    let onReset: () -> Void

    @Environment(\.colorScheme) private var scheme

    /// Effective on/off shown by the toggle: override if set, else global.
    private var effective: Bool {
        switch override {
        case "on":  return true
        case "off": return false
        default:    return globalValue
        }
    }

    private var isOverride: Bool { override == "on" || override == "off" }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .center, spacing: JunoUI.Spacing.m) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .junoType(.bodyEmphasis)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text(subtitle)
                        .junoType(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: JunoUI.Spacing.m)
                Toggle("", isOn: Binding(
                    get: { effective },
                    set: { newValue in onToggle(newValue) }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }
            inheritanceLine
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private var inheritanceLine: some View {
        HStack(spacing: 8) {
            if isOverride {
                Text("Just for this app")
                    .junoType(.caption)
                    .foregroundStyle(JunoDesignTokens.accent)
                Text("·")
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.6))
                Button("Reset to global") { onReset() }
                    .buttonStyle(.plain)
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .underline()
            } else {
                Text(globalLoaded
                     ? "Following your global setting"
                     : "Following your global setting…")
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }
            Spacer(minLength: 0)
        }
    }
}

// MARK: - Custom dropdown controls

/// App picker with a custom-styled trigger and a Popover-driven picker. Shows
/// app icon + name, never a bundle id, and never a chevron-only NSPopUpButton.
private struct AppPickerButton: View {
    @Binding var selection: String
    let runningApps: [(bundle: String, name: String, icon: NSImage?)]
    let resolvedDisplay: (name: String, icon: NSImage?)?
    let onPick: (String) -> Void
    let onAddCustom: () -> Void

    @State private var open = false
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Button { open.toggle() } label: { triggerLabel }
            .buttonStyle(.plain)
            .popover(isPresented: $open, arrowEdge: .bottom) { popoverContent }
    }

    private var triggerLabel: some View {
        HStack(spacing: JunoUI.Spacing.m) {
            iconView
            VStack(alignment: .leading, spacing: 2) {
                Text(resolvedDisplay?.name ?? "Pick an app")
                    .junoType(.bodyEmphasis)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(resolvedDisplay == nil
                     ? "Choose any app you use often."
                     : "Tap to change which app you’re setting up.")
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }
            Spacer(minLength: JunoUI.Spacing.s)
            Image(systemName: "chevron.up.chevron.down")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.7))
        }
        .padding(.horizontal, JunoUI.Spacing.m)
        .padding(.vertical, JunoUI.Spacing.s)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme).opacity(0.45), lineWidth: 0.5)
        )
        .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private var iconView: some View {
        let size: CGFloat = 36
        return Group {
            if let icon = resolvedDisplay?.icon {
                Image(nsImage: icon).resizable().frame(width: size, height: size).cornerRadius(size * 0.22)
            } else {
                RoundedRectangle(cornerRadius: size * 0.22, style: .continuous)
                    .fill(JunoTheme.cardBackground(scheme))
                    .frame(width: size, height: size)
                    .overlay(
                        Image(systemName: "app.badge")
                            .font(.system(size: size * 0.45, weight: .regular))
                            .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.6))
                    )
            }
        }
    }

    private var popoverContent: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: 0) {
                    ForEach(runningApps, id: \.bundle) { row in
                        Button {
                            onPick(row.bundle)
                            open = false
                        } label: {
                            HStack(spacing: 10) {
                                if let icon = row.icon {
                                    Image(nsImage: icon).resizable().frame(width: 20, height: 20).cornerRadius(4)
                                }
                                Text(row.name)
                                    .junoType(.body)
                                    .foregroundStyle(JunoTheme.primaryText(scheme))
                                Spacer()
                                if row.bundle == selection {
                                    Image(systemName: "checkmark")
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundStyle(JunoDesignTokens.accent)
                                }
                            }
                            .contentShape(Rectangle())
                            .padding(.horizontal, JunoUI.Spacing.m)
                            .padding(.vertical, 7)
                        }
                        .buttonStyle(JunoPopoverRowButtonStyle())
                        .junoNoFocusEffect()
                    }
                }
            }
            .frame(maxHeight: 320)
            Divider()
            Button {
                onAddCustom()
                open = false
            } label: {
                HStack {
                    Image(systemName: "plus")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(JunoDesignTokens.accent)
                    Text("Add an app that isn’t open")
                        .junoType(.body)
                        .foregroundStyle(JunoDesignTokens.accent)
                    Spacer()
                }
                .contentShape(Rectangle())
                .padding(.horizontal, JunoUI.Spacing.m)
                .padding(.vertical, 9)
            }
            .buttonStyle(JunoPopoverRowButtonStyle())
                        .junoNoFocusEffect()
        }
        .frame(width: 280)
    }
}

/// Writing-style picker with a custom trigger that shows the chosen mode's
/// title + one-line summary, and a Popover with rich rows.
private struct WritingStyleButton: View {
    let selectedId: String
    let modes: [(String, String)]
    let onPick: (String) -> Void

    @State private var open = false
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Button { open.toggle() } label: { triggerLabel }
            .buttonStyle(.plain)
            .popover(isPresented: $open, arrowEdge: .bottom) { popoverContent }
    }

    private var triggerLabel: some View {
        HStack(spacing: JunoUI.Spacing.m) {
            VStack(alignment: .leading, spacing: 2) {
                Text(triggerTitle)
                    .junoType(.bodyEmphasis)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(triggerSubtitle)
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .lineLimit(1)
            }
            Spacer(minLength: JunoUI.Spacing.s)
            Image(systemName: "chevron.up.chevron.down")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.7))
        }
        .padding(.horizontal, JunoUI.Spacing.m)
        .padding(.vertical, JunoUI.Spacing.s)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme).opacity(0.45), lineWidth: 0.5)
        )
        .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private var triggerTitle: String {
        if let row = modes.first(where: { $0.0 == selectedId }) { return row.1 }
        return "Pick a style"
    }

    private var triggerSubtitle: String {
        if selectedId.isEmpty { return "Juno’s default style is used until you pick one." }
        return writingStyleSummary(for: selectedId)
    }

    private var popoverContent: some View {
        ScrollView {
            VStack(spacing: 0) {
                ForEach(modes, id: \.0) { row in
                    Button {
                        onPick(row.0)
                        open = false
                    } label: {
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: row.0 == selectedId ? "largecircle.fill.circle" : "circle")
                                .font(.system(size: 14, weight: .regular))
                                .foregroundStyle(row.0 == selectedId ? JunoDesignTokens.accent : JunoTheme.secondaryText(scheme).opacity(0.7))
                                .padding(.top, 2)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(row.1)
                                    .junoType(.bodyEmphasis)
                                    .foregroundStyle(JunoTheme.primaryText(scheme))
                                Text(writingStyleSummary(for: row.0))
                                    .junoType(.caption)
                                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            Spacer(minLength: 0)
                        }
                        .contentShape(Rectangle())
                        .padding(.horizontal, JunoUI.Spacing.m)
                        .padding(.vertical, 9)
                    }
                    .buttonStyle(JunoPopoverRowButtonStyle())
                        .junoNoFocusEffect()
                }
            }
        }
        .frame(width: 320, height: min(CGFloat(modes.count) * 56 + 8, 360))
    }
}
