import AppKit
import AVFoundation
import ServiceManagement
import SwiftUI

// Premium, organized Settings surface (single curated scroll + Advanced drill-in).

private enum JunoPendingStorageAction: Identifiable, Equatable {
    case clearHistory
    case deleteRecordings
    case cleanup

    var id: String {
        switch self {
        case .clearHistory: return "clear_history"
        case .deleteRecordings: return "delete_recordings"
        case .cleanup: return "cleanup"
        }
    }

    var title: String {
        switch self {
        case .clearHistory: return "Clear all dictation history?"
        case .deleteRecordings: return "Delete all retained recordings?"
        case .cleanup: return "Clean up old storage now?"
        }
    }

    var message: String {
        switch self {
        case .clearHistory:
            return "This removes saved history entries from this Mac. It cannot be undone."
        case .deleteRecordings:
            return "This removes retained audio recordings used for replay and troubleshooting. It cannot be undone."
        case .cleanup:
            return "Juno will apply your current retention settings and remove anything older than those limits."
        }
    }

    var buttonTitle: String {
        switch self {
        case .clearHistory: return "Clear History"
        case .deleteRecordings: return "Delete Recordings"
        case .cleanup: return "Clean Up Now"
        }
    }
}

struct JunoSettingsView: View {
    @ObservedObject private var perms = JunoPermissionMonitor.shared
    @ObservedObject private var updater = JunoUpdater.shared
    @ObservedObject var setup: JunoSetupModel
    @StateObject private var retention = JunoRetentionSettingsModel()
    @StateObject private var privacy = JunoPrivacySettingsModel()

    @State private var selectedShortcut: JunoShortcutPreference = JunoShortcutPreference.stored
    @State private var launchAtLoginEnabled = false
    @State private var suppressLaunchAtLoginOnChange = true
    @State private var launchAtLoginError: String?
    @State private var displayNameDraft: String = JunoUserDefaults.preferredDisplayName ?? ""

    @State private var appearancePreference = JunoUserDefaults.appearancePreference
    @State private var showInDock = JunoUserDefaults.showInDock
    @State private var micProcessing = JunoUserDefaults.micVoiceProcessingEnabled
    @State private var hudPosition = JunoUserDefaults.hudPosition
    @State private var hudLiveTranscriptions = JunoUserDefaults.hudLiveTranscriptionsEnabled
    @State private var previewEligibility = JunoPreviewEligibility.current
    @State private var hudSounds = JunoUserDefaults.hudOpenSoundEnabled
    @State private var hudShowDoneRow = JunoUserDefaults.hudShowDoneRowEnabled
    @State private var pauseSensitivitySeconds = JunoUserDefaults.pauseSensitivitySeconds
    @State private var languageMode = JunoUserDefaults.languageMode
    @State private var screenContextEnabled = JunoUserDefaults.screenContextEnabled
    @State private var screenContextPermissionGranted = JunoScreenContextAccess.permissionGranted
    @State private var showModelDetails = false
    @State private var developerMode = JunoUserDefaults.developerModeEnabled
    @State private var pendingStorageAction: JunoPendingStorageAction?

    // Voice Actions state lives entirely inside ``JunoVoiceActionsBanner``
    // so this view doesn't have to coordinate three separate flags
    // (master toggle, TCC, signature) and risk drift between them. The
    // banner is the single source of truth for the feature on this page.

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        NavigationStack {
            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: JunoUI.Spacing.m) {
                    JunoPageHeader(
                        eyebrow: "Settings",
                        title: "Make Juno feel right on this Mac",
                        subtitle: "Tune listening, writing, memory, storage, and app behavior without leaving Juno."
                    )
                    .padding(.bottom, JunoUI.Spacing.xs)

                    // Voice Actions are managed from the dedicated Actions
                    // sidebar page (catalog, permissions, examples). Keep
                    // a slim pointer here so users coming to Settings
                    // looking for Voice Actions still find them.
                    JunoSettingsActionsPointer()
                    screenContextSettingsCard

                    settingsCard(
                        title: "General",
                        subtitle: "The everyday controls most people reach for first."
                    ) {
                        VStack(spacing: 10) {
                            settingsRow(
                                title: "Appearance",
                                subtitle: "Choose a fixed look, or match your Mac's Appearance setting.",
                                trailing: {
                                    AppearancePopoverPicker(selection: $appearancePreference) { newValue in
                                        JunoUserDefaults.appearancePreference = newValue
                                    }
                                }
                            )

                            Divider().opacity(0.25)

                            settingsRow(
                                title: "Show in Dock",
                                subtitle: "Turn off to use Juno from the menu bar only — reopen the window from the Juno menu bar icon.",
                                trailing: {
                                    Toggle("", isOn: $showInDock)
                                        .labelsHidden()
                                        .onChange(of: showInDock) { newValue in
                                            JunoUserDefaults.showInDock = newValue
                                            Task { @MainActor in JunoDockVisibility.applyCurrent() }
                                        }
                                }
                            )

                            Divider().opacity(0.25)

                            settingsRow(
                                title: "Keep recordings for",
                                subtitle: "Used for replay, re-run, and Insert again. Recent sessions are always kept for at least 7 days so you can recover from a failed paste — even when this is set to Off.",
                                trailing: {
                                    RetentionPopoverPicker(
                                        selection: $retention.audioChoice,
                                        kind: .audio,
                                        isDisabled: !retention.brokerReachable
                                    ) { _ in
                                        retention.persistAndCleanup()
                                    }
                                }
                            )

                            Divider().opacity(0.25)

                            VStack(alignment: .leading, spacing: 8) {
                                settingsRow(
                                    title: "Live transcriptions",
                                    subtitle: "Show what you're saying inside the HUD as you speak. When off, the HUD collapses to a tiny waveform.",
                                    trailing: {
                                        Toggle("", isOn: Binding(
                                            get: { hudLiveTranscriptions },
                                            set: { newValue in
                                                setLiveTranscriptionPreview(newValue)
                                            }
                                        ))
                                        .labelsHidden()
                                        .disabled(!previewEligibility.isEligible)
                                    }
                                )

                                if let message = livePreviewResourceMessage {
                                    Label(
                                        message,
                                        systemImage: previewEligibility.isEligible
                                            ? "exclamationmark.triangle.fill"
                                            : "lock.fill"
                                    )
                                    .font(.caption2)
                                    .foregroundStyle(
                                        previewEligibility.isEligible
                                            ? JunoDesignTokens.danger
                                            : JunoTheme.secondaryText(scheme)
                                    )
                                    .fixedSize(horizontal: false, vertical: true)
                                }
                            }

                            Divider().opacity(0.25)

                            settingsRow(
                                title: "HUD position",
                                subtitle: "Top is the default. Choose Bottom if it conflicts with your menu bar or notch.",
                                trailing: {
                                    HUDPositionPopoverPicker(selection: $hudPosition) { newValue in
                                        JunoUserDefaults.hudPosition = newValue
                                    }
                                }
                            )
                        }
                    }

                    settingsCard(
                        title: "Dictation",
                        subtitle: "How Juno starts, confirms, and stays visible while you speak."
                    ) {
                        settingsRow(
                            title: "Shortcut",
                            subtitle: "Press to start · press again to finish",
                            trailing: {
                                ShortcutPopoverPicker(selection: $selectedShortcut) { newValue in
                                    JunoShortcutPreference.applyShortcutSelection(newValue)
                                }
                            }
                        )

                        if selectedShortcut == .fn {
                            HStack(alignment: .top, spacing: 8) {
                                Image(systemName: "info.circle.fill")
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundStyle(JunoDesignTokens.accent)
                                Text(JunoShortcutPreference.fnGlobeConflictNote)
                                    .font(.caption)
                                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .padding(.top, 4)
                        }

                        Divider().opacity(0.25)

                        settingsRow(
                            title: "HUD sounds",
                            subtitle: "Soft cue when the HUD opens and closes",
                            trailing: {
                                Toggle("", isOn: $hudSounds)
                                    .labelsHidden()
                                    .onChange(of: hudSounds) { newValue in
                                        // Single toggle now governs both
                                        // open and close cues. Keep both
                                        // legacy keys mirrored so older
                                        // call sites stay consistent.
                                        JunoUserDefaults.hudOpenSoundEnabled = newValue
                                        JunoUserDefaults.hudDelightSoundEnabled = newValue
                                    }
                            }
                        )

                        Divider().opacity(0.25)

                        settingsRow(
                            title: "Pause sensitivity",
                            subtitle: "How long Juno waits after a pause before it finalizes what you said.",
                            trailing: {
                                HStack(spacing: 8) {
                                    Slider(value: $pauseSensitivitySeconds, in: 0.8...3.0, step: 0.1)
                                        .frame(width: 150)
                                        .onChange(of: pauseSensitivitySeconds) { newValue in
                                            JunoUserDefaults.pauseSensitivitySeconds = newValue
                                        }
                                    Text(String(format: "%.1fs", pauseSensitivitySeconds))
                                        .font(.caption.monospacedDigit())
                                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                                        .frame(width: 36, alignment: .trailing)
                                }
                            }
                        )

                        Divider().opacity(0.25)

                        settingsRow(
                            title: "Show word count after dictation",
                            subtitle: "Brief \"+N words\" confirmation as the HUD fades away",
                            trailing: {
                                Toggle("", isOn: $hudShowDoneRow)
                                    .labelsHidden()
                                    .onChange(of: hudShowDoneRow) { newValue in
                                        JunoUserDefaults.hudShowDoneRowEnabled = newValue
                                    }
                            }
                        )

                        Divider().opacity(0.25)

                        settingsRow(
                            title: "Home greeting",
                            subtitle: "Optional name used on Home",
                            trailing: {
                                TextField("Your name", text: $displayNameDraft)
                                    .textFieldStyle(.roundedBorder)
                                    .focusEffectDisabled()
                                    .frame(maxWidth: 220)
                                    .onChange(of: displayNameDraft) { newValue in
                                        let t = newValue.trimmingCharacters(in: .whitespacesAndNewlines)
                                        JunoUserDefaults.preferredDisplayName = t.isEmpty ? nil : t
                                    }
                            }
                        )
                    }

                    settingsCard(
                        title: "Audio input",
                        subtitle: "Keep defaults unless your microphone sounds clipped, hollow, or too processed."
                    ) {
                        settingsRow(
                            title: "Mic processing",
                            subtitle: "Reduces background noise and evens out volume. Try turning this off if dictation sounds garbled.",
                            trailing: {
                                Toggle("", isOn: $micProcessing)
                                    .labelsHidden()
                                    .onChange(of: micProcessing) { newValue in
                                        JunoUserDefaults.micVoiceProcessingEnabled = newValue
                                    }
                            }
                        )
                    }

                    settingsCard(
                        title: "Writing & language",
                        subtitle: "Choose how Juno interprets speech."
                    ) {
                        settingsRow(
                            title: "Language",
                            subtitle: "Auto works best for most people",
                            trailing: {
                                GenericGlyphPopoverPicker(
                                    selection: $languageMode,
                                    options: JunoLanguagePickerOptions.all
                                ) { newValue in
                                    JunoUserDefaults.languageMode = newValue
                                    persistLanguageEnvironment()
                                }
                            }
                        )
                    }

                    settingsCard(
                        title: "Storage",
                        subtitle: "Control what is kept locally for replay, review, and troubleshooting."
                    ) {
                        VStack(spacing: 12) {
                            if !retention.brokerReachable {
                                Text("Voice engine not connected — storage controls are unavailable until Juno reconnects.")
                                    .font(.caption)
                                    .foregroundStyle(JunoDesignTokens.danger)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            settingsRow(
                                title: "Keep history for",
                                subtitle: "Shown in History on this Mac",
                                trailing: {
                                    RetentionPopoverPicker(
                                        selection: $retention.historyChoice,
                                        kind: .history,
                                        isDisabled: !retention.brokerReachable
                                    ) { _ in
                                        retention.persistAndCleanup()
                                    }
                                }
                            )

                            if let stats = retention.storageSummaryLine {
                                JunoHairlineRule(.faint)
                                Text(stats)
                                    .font(.caption)
                                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                                    .fixedSize(horizontal: false, vertical: true)
                            }

                            if retention.storageLogDir != nil {
                                Button {
                                    retention.openStorageLogDirInFinder()
                                } label: {
                                    Label("Reveal storage folder", systemImage: "folder")
                                        .font(.caption)
                                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                                }
                                .buttonStyle(.plain)
                                .disabled(!retention.brokerReachable)
                            }

                            Divider().opacity(0.25)

                            HStack(spacing: 10) {
                                Button("Clear history…") { pendingStorageAction = .clearHistory }
                                    .buttonStyle(.bordered)
                                    .controlSize(.small)
                                    .disabled(!retention.brokerReachable || retention.isBusy)

                                Button("Delete recordings…") { pendingStorageAction = .deleteRecordings }
                                    .buttonStyle(.bordered)
                                    .controlSize(.small)
                                    .disabled(!retention.brokerReachable || retention.isBusy)

                                Spacer()

                                Button(retention.isBusy ? "Working…" : "Clean up now") {
                                    pendingStorageAction = .cleanup
                                }
                                    .junoPrimaryActionButton()
                                    .disabled(!retention.brokerReachable || retention.isBusy)
                            }

                            if let err = retention.inlineError {
                                Text(err)
                                    .font(.caption2)
                                    .foregroundStyle(JunoDesignTokens.danger)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .padding(.top, 2)
                            }
                        }
                    }

                    settingsCard(
                        title: "Privacy & learning",
                        subtitle: "Control the local context Juno can use when writing."
                    ) {
                        Text("Context helps Juno spell names, use the right tone, and edit selected text. You can turn it off globally or per app.")
                            .font(.caption)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .fixedSize(horizontal: false, vertical: true)
                        if !privacy.brokerReachable {
                            Text("Voice engine not connected — these toggles can't save right now.")
                                .font(.caption)
                                .foregroundStyle(JunoDesignTokens.danger)
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(.top, 2)
                        }
                        VStack(spacing: 10) {
                            privacyToggle("Smart Context", "Use safe local app context while writing", $privacy.smartContext)
                            Divider().opacity(0.25)
                            screenContextSettingsRow
                            Divider().opacity(0.25)
                            privacyToggle("Use selected text", "Let Juno edit highlighted text", $privacy.useSelectedText)
                            Divider().opacity(0.25)
                            privacyToggle("Use current app/field", "Use focused field context when available", $privacy.useFocusedText)
                            Divider().opacity(0.25)
                            privacyToggle("Use app/window title", "Helps choose writing behavior", $privacy.useWindowTitle)
                            Divider().opacity(0.25)
                            privacyToggle("Use recent clipboard", "Off by default", $privacy.useClipboard)
                            Divider().opacity(0.25)
                            privacyToggle("Learn from corrections", "Improve words you manually fix", $privacy.learnFromCorrections)
                        }
                        if let err = privacy.inlineError {
                            Text(err)
                                .font(.caption2)
                                .foregroundStyle(JunoDesignTokens.danger)
                        }
                    }

                    settingsCard(
                        title: "Permissions",
                        subtitle: "macOS owns these permissions; Juno shows the current state and opens the right place to fix them."
                    ) {
                        Text("These permissions are managed by macOS. Juno can only show their status here.")
                            .font(.caption)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .fixedSize(horizontal: false, vertical: true)

                        VStack(spacing: 8) {
                            permCard(
                                icon: "mic",
                                label: "Microphone",
                                detail: perms.micStatusLabel + " — Required for dictation.",
                                ok: perms.micStatus == .authorized,
                                primaryTitle: micPrimaryTitle,
                                primaryAction: micPrimaryAction,
                                showSecondarySettings: false,
                                settingsAction: { perms.openMicSettings() }
                            )
                            permCard(
                                icon: "hand.raised",
                                label: "Accessibility",
                                detail: perms.axGranted ? "Granted — Juno can insert text." : "Required to read the focused field and insert text.",
                                ok: perms.axGranted,
                                primaryTitle: "Open Accessibility",
                                primaryAction: {
                                    perms.openAccessibilitySettings()
                                },
                                showSecondarySettings: false,
                                settingsAction: { perms.openAXSettings() }
                            )
                            permCard(
                                icon: "viewfinder",
                                label: "Visible screen text",
                                detail: perms.screenRecordingStatusLabel + " — Optional local OCR for names and code terms.",
                                ok: perms.screenContextEnabled && perms.screenRecordingGranted,
                                primaryTitle: screenRecordingPrimaryTitle,
                                primaryAction: screenRecordingPrimaryAction,
                                showSecondarySettings: false,
                                settingsAction: { perms.openScreenRecordingSettings() }
                            )
                        }

                        if !perms.axGranted {
                            Text("After enabling Accessibility, this status will refresh within a few seconds.")
                                .font(.caption2)
                                .foregroundStyle(JunoTheme.secondaryText(scheme))
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(.top, 2)
                        }

                        HStack(spacing: 10) {
                            Button("Check again") { perms.refresh() }
                                .font(.caption)
                                .foregroundStyle(JunoTheme.secondaryText(scheme))

                            Spacer()
                        }
                        .padding(.top, 4)
                    }

                    settingsCard(
                        title: "Models",
                        subtitle: "Local speech and writing models that power Juno."
                    ) {
                        modelRuntimeSection
                    }

                    settingsCard(
                        title: "Updates & app",
                        subtitle: "Launch behavior, software updates, and developer tools."
                    ) {
                        VStack(spacing: 12) {
                            if updater.isConfigured {
                                if updater.updateAvailable {
                                    JunoUpdateAvailableBanner(
                                        version: updater.latestVersion,
                                        onInstall: { updater.checkForUpdates() }
                                    )
                                    .transition(.opacity.combined(with: .move(edge: .top)))
                                }

                                settingsRow(
                                    title: "Software updates",
                                    subtitle: updater.lastError != nil
                                        ? "Couldn't check for updates — try again."
                                        : updater.statusLine,
                                    trailing: {
                                        Button(updater.updateAvailable ? "Install…" : "Check") {
                                            updater.checkForUpdates()
                                        }
                                        .buttonStyle(.bordered)
                                        .controlSize(.small)
                                        .disabled(!updater.canCheckForUpdates)
                                    }
                                )

                                JunoHairlineRule(.faint)

                                settingsRow(
                                    title: "Check automatically",
                                    subtitle: "Look for new Juno releases in the background",
                                    trailing: {
                                        Toggle("", isOn: Binding(
                                            get: { updater.automaticallyChecksForUpdates },
                                            set: { updater.setAutomaticallyChecksForUpdates($0) }
                                        ))
                                        .labelsHidden()
                                    }
                                )

                                JunoHairlineRule(.faint)

                                settingsRow(
                                    title: "Download updates automatically",
                                    subtitle: "Prepare updates in the background; Juno still asks before relaunching",
                                    trailing: {
                                        Toggle("", isOn: Binding(
                                            get: { updater.automaticallyDownloadsUpdates },
                                            set: { updater.setAutomaticallyDownloadsUpdates($0) }
                                        ))
                                        .labelsHidden()
                                        .disabled(!updater.allowsAutomaticUpdates)
                                    }
                                )

                                JunoHairlineRule(.faint)
                            }

                            settingsRow(
                                title: "Launch on login",
                                subtitle: launchAtLoginError ?? "Starts quietly in the background after you sign in",
                                trailing: {
                                    Toggle("", isOn: $launchAtLoginEnabled)
                                        .labelsHidden()
                                        .onChange(of: launchAtLoginEnabled) { newValue in
                                            guard !suppressLaunchAtLoginOnChange else { return }
                                            setLaunchAtLogin(newValue)
                                        }
                                }
                            )
                            if launchAtLoginError != nil {
                                Text("Juno couldn't update the login item. Try again, or set this manually in System Settings → General → Login Items.")
                                    .font(.caption2)
                                    .foregroundStyle(JunoDesignTokens.danger)
                                    .fixedSize(horizontal: false, vertical: true)
                            }

                            Divider().opacity(0.25)

                            settingsRow(
                                title: "Developer mode",
                                subtitle: "Shows extra diagnostics for troubleshooting",
                                trailing: {
                                    Toggle("", isOn: $developerMode)
                                        .labelsHidden()
                                        .onChange(of: developerMode) { newValue in
                                            JunoUserDefaults.developerModeEnabled = newValue
                                        }
                                }
                            )
                        }
                    }

                    if developerMode {
                        JunoSectionLabel(text: "Developer")
                            .padding(.top, 2)
                        Text("Rare options and diagnostics. Hidden when Developer mode is off.")
                            .font(.caption)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .fixedSize(horizontal: false, vertical: true)
                        JunoAdvancedSettingsView()
                    }

                    // Always visible (not developer-gated): end users are
                    // exactly who needs a complete uninstall.
                    JunoUninstallSettingsCard()
                }
                .junoDetailPagePadding()
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .contentShape(Rectangle())
            .toolbar(.hidden, for: .automatic)
            .overlay(alignment: .top) {
                // Pinned instead of part of the ScrollView, so broker-save
                // failures remain visible when the user is editing lower cards.
                JunoSettingsToastBanner()
                    .padding(.top, 10)
            }
        }
        .onAppear {
            displayNameDraft = JunoUserDefaults.preferredDisplayName ?? ""
            pauseSensitivitySeconds = JunoUserDefaults.pauseSensitivitySeconds
            screenContextEnabled = JunoUserDefaults.screenContextEnabled
            refreshPreviewEligibility()
            refreshScreenContextPermission()
            perms.refresh()
            retention.refresh()
            privacy.refresh()
            updater.startIfConfigured()
            updater.refreshState()
            suppressLaunchAtLoginOnChange = true
            launchAtLoginEnabled = currentLaunchAtLoginEnabled()
            suppressLaunchAtLoginOnChange = false
        }
        .confirmationDialog(
            pendingStorageAction?.title ?? "Confirm storage action",
            isPresented: Binding(
                get: { pendingStorageAction != nil },
                set: { if !$0 { pendingStorageAction = nil } }
            ),
            titleVisibility: .visible
        ) {
            if let action = pendingStorageAction {
                Button(action.buttonTitle, role: action == .cleanup ? nil : .destructive) {
                    performStorageAction(action)
                    pendingStorageAction = nil
                }
                Button("Cancel", role: .cancel) {
                    pendingStorageAction = nil
                }
            }
        } message: {
            if let action = pendingStorageAction {
                Text(action.message)
            }
        }
    }

    // MARK: Model / runtime card

    private var setupLaneItems: [JunoSetupLaneViewModel] {
        JunoSetupPresentation.laneItems(from: setup).filter { $0.required || $0.role != .writer }
    }

    private var modelRuntimeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: setup.overallReady ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(setup.overallReady ? .green : .orange)
                Text(setup.readinessLabel)
                    .font(.system(.callout, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Spacer()
                if setup.canInstall {
                    Button("Install") { setup.triggerInstall() }
                        .junoPrimaryActionButton()
                } else if setup.installState == "downloading" {
                    ProgressView().controlSize(.small)
                } else if setup.canRepair {
                    Button("Repair") { setup.triggerRepair() }
                        .buttonStyle(.bordered).controlSize(.small)
                }
            }

            VStack(spacing: 8) {
                ForEach(setupLaneItems) { lane in
                    HStack(spacing: 8) {
                        Image(systemName: lane.ready ? "checkmark.circle.fill" : "arrow.down.circle")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(lane.ready ? JunoDesignTokens.meadow : JunoTheme.secondaryText(scheme))
                            .frame(width: 14)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(lane.title)
                                .font(.system(.callout, design: .rounded).weight(.semibold))
                                .foregroundStyle(JunoTheme.primaryText(scheme))
                            Text(lane.role == .writer ? "Improves rewriting and styles" : "Runs locally for dictation")
                                .font(.system(size: 11, weight: .medium, design: .rounded))
                                .foregroundStyle(JunoTheme.secondaryText(scheme))
                                .lineLimit(1)
                        }
                        Spacer()
                        Text(lane.ready ? "Ready" : "Missing")
                            .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                            .foregroundStyle(lane.ready ? JunoTheme.secondaryText(scheme) : JunoDesignTokens.danger)
                    }
                }
            }

            if setup.writerRequired && !setup.writerModelCached {
                HStack(spacing: 8) {
                    Image(systemName: "pencil.slash")
                        .foregroundStyle(.orange)
                        .font(.caption)
                    Text("Writing styles use basic formatting until the writing model finishes downloading.")
                        .font(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.vertical, 4)
            }

            if !setup.checks.isEmpty {
                Divider().opacity(0.3)
                DisclosureGroup(isExpanded: $showModelDetails) {
                    VStack(spacing: 8) {
                        ForEach(setup.checks) { check in
                            HStack(spacing: 8) {
                                Image(systemName: check.ok ? "checkmark" : "xmark")
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(check.ok ? Color.green : Color.red)
                                    .frame(width: 14)
                                Text(friendlyModelName(check.name))
                                    .font(.system(.callout, design: .rounded))
                                    .foregroundStyle(JunoTheme.primaryText(scheme))
                                if let lane = setupLaneItems.first(where: { laneForCheckName(check.name) == $0.role }) {
                                    Text(lane.modelName)
                                        .font(.system(size: 10, weight: .medium, design: .rounded))
                                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                                        .lineLimit(1)
                                }
                                Spacer()
                                Text(check.ok ? "Ready" : "Missing")
                                    .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                                    .foregroundStyle(check.ok ? JunoTheme.secondaryText(scheme) : .red)
                            }
                        }
                    }
                    .padding(.top, 6)
                } label: {
                    Text("Model details")
                        .junoType(.label)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
            }
        }
    }

    private func friendlyModelName(_ raw: String) -> String {
        switch raw {
        case "preview_model": return "Fast preview model"
        case "final_model":   return "Final dictation model"
        case "writer_model":  return "Writing engine"
        default: return raw.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    private func laneForCheckName(_ raw: String) -> JunoSetupLaneRole {
        switch raw {
        case "preview_model":
            return .preview
        case "final_model":
            return .final
        default:
            return .writer
        }
    }

    // MARK: Permissions

    private var micPrimaryTitle: String {
        switch perms.micStatus {
        case .notDetermined: return "Allow microphone"
        case .denied, .restricted: return "Open Microphone privacy"
        case .authorized: return "Granted"
        @unknown default: return "Open Microphone privacy"
        }
    }

    private func micPrimaryAction() {
        switch perms.micStatus {
        case .notDetermined: perms.requestMic()
        default: perms.openMicSettings()
        }
    }

    private var screenRecordingPrimaryTitle: String {
        if perms.screenContextEnabled && perms.screenRecordingGranted {
            return "Granted"
        }
        return "Open Screen Recording"
    }

    private func screenRecordingPrimaryAction() {
        if perms.screenContextEnabled && perms.screenRecordingGranted {
            perms.openScreenRecordingSettings()
        } else {
            perms.requestScreenRecording()
        }
    }

    private func permCard(icon: String, label: String, detail: String, ok: Bool,
                          primaryTitle: String,
                          primaryAction: @escaping () -> Void,
                          showSecondarySettings: Bool,
                          settingsAction: @escaping () -> Void) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .light))
                .foregroundStyle(ok ? Color.green : JunoDesignTokens.muted)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 3) {
                Text(label)
                    .font(.system(.callout, design: .rounded).weight(.semibold))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(detail)
                    .font(.caption).foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            if !ok {
                VStack(alignment: .trailing, spacing: 4) {
                    Button(primaryTitle) { primaryAction() }
                        .junoPrimaryActionButton()
                    if showSecondarySettings {
                        Button("Open Settings") { settingsAction() }
                            .buttonStyle(.bordered).controlSize(.small)
                    }
                }
            } else {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.system(size: 16))
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(ok ? Color.green.opacity(0.06) : JunoDesignTokens.accent.opacity(0.05))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(ok ? Color.green.opacity(0.18) : JunoDesignTokens.border.opacity(0.5), lineWidth: 0.5)
        )
    }

    // MARK: Card + row helpers

    private func settingsCard<Content: View>(
        title: String,
        subtitle: String? = nil,
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        JunoPreferenceSection(title: title, subtitle: subtitle) {
            content()
        }
    }

    private func settingsRow<Trailing: View>(
        title: String,
        subtitle: String? = nil,
        @ViewBuilder trailing: @escaping () -> Trailing
    ) -> some View {
        JunoPreferenceRow(title: title, subtitle: subtitle) {
            trailing()
        }
    }

    private var screenContextSettingsRow: some View {
        settingsRow(
            title: "Visible screen text",
            subtitle: screenContextSubtitle,
            trailing: {
                if screenContextEnabled && screenContextPermissionGranted {
                    Button("Turn off") {
                        setScreenContextEnabled(false)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                } else {
                    Button("Open Screen Recording") {
                        setScreenContextEnabled(true)
                    }
                    .junoPrimaryActionButton()
                }
            }
        )
    }

    private var screenContextSettingsCard: some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: "viewfinder")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(JunoDesignTokens.accent)
                .frame(width: 22)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text("Visible screen text")
                        .font(.system(.subheadline, design: .rounded).weight(.semibold))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text(screenContextStatusLabel)
                        .font(.system(size: 10, weight: .semibold, design: .rounded))
                        .foregroundStyle(screenContextStatusColor)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2)
                        .background(
                            Capsule(style: .continuous)
                                .fill(screenContextStatusColor.opacity(scheme == .dark ? 0.18 : 0.11))
                        )
                }
                Text(screenContextTopCardSubtitle)
                    .font(.system(.footnote, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 12)

            if screenContextEnabled && screenContextPermissionGranted {
                Button("Turn off") {
                    setScreenContextEnabled(false)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                Button("Open Screen Recording") {
                    requestScreenContextPermission()
                }
                .junoPrimaryActionButton()
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .premiumCard()
        .onAppear {
            refreshScreenContextPermission()
        }
    }

    private var screenContextStatusLabel: String {
        if !screenContextEnabled { return "Off" }
        return screenContextPermissionGranted ? "On" : "Needs approval"
    }

    private var screenContextStatusColor: Color {
        if !screenContextEnabled { return JunoTheme.secondaryText(scheme) }
        return screenContextPermissionGranted ? JunoDesignTokens.meadow : Color.orange
    }

    private var screenContextTopCardSubtitle: String {
        if !screenContextEnabled {
            return "Optional local OCR for names, code terms, and product words already visible while you dictate."
        }
        if screenContextPermissionGranted {
            return "Juno can prioritize visible terms while dictating. Text is not stored and does not leave this Mac."
        }
        return "Juno is ready to be turned on in macOS Screen Recording so visible terms can be read locally while dictating."
    }

    private var screenContextSubtitle: String {
        if !screenContextEnabled {
            return "Optional. Off by default; enable it here when you want visible names and code terms prioritized."
        }
        if screenContextPermissionGranted {
            return "Reads visible names and code terms with on-device OCR while you dictate. Never stored, never leaves this Mac."
        }
        return "Turn on Juno in macOS Screen Recording. Used only for on-device OCR while dictating; never stored or sent."
    }

    private func setScreenContextEnabled(_ enabled: Bool) {
        screenContextEnabled = enabled
        JunoUserDefaults.screenContextEnabled = enabled
        if enabled {
            refreshScreenContextPermission()
            if !screenContextPermissionGranted {
                requestScreenContextPermission()
            }
        } else {
            JunoScreenTermHarvester.shared.deactivate()
            refreshScreenContextPermission()
        }
    }

    private func refreshScreenContextPermission() {
        screenContextPermissionGranted = JunoScreenContextAccess.permissionGranted
    }

    private func requestScreenContextPermission() {
        screenContextEnabled = true
        JunoUserDefaults.screenContextEnabled = true
        JunoScreenContextAccess.requestFromExplicitUserAction { granted in
            screenContextPermissionGranted = granted
        }
    }

    private func privacyToggle(_ title: String, _ subtitle: String, _ binding: Binding<Bool>) -> some View {
        settingsRow(
            title: title,
            subtitle: subtitle,
            trailing: {
                Toggle("", isOn: binding)
                    .labelsHidden()
                    .disabled(!privacy.brokerReachable)
                    .onChange(of: binding.wrappedValue) { _ in
                        privacy.persist()
                    }
            }
        )
    }

    private func persistLanguageEnvironment() {
        JunoBroker.postJSON(
            path: "api/broker/settings/language_environment",
            payload: ["language_mode": languageMode]
        ) { obj in
            let ok = (obj["ok"] as? Bool) ?? false
            if !ok {
                let msg = (obj["error"] as? String) ?? "Could not save language settings"
                JunoSettingsToastCenter.shared.report(msg)
            }
        }
    }

    /// Mirror the macOS-side toggle into the broker so the engine can suppress
    /// per-utterance preview-lane decoding on the next dictation session. The
    /// macOS HUD layout flips immediately off `JunoUserDefaults`; the engine
    /// gate is read at session start.
    private func persistLiveCaptionEnabled(_ enabled: Bool, reportFailures: Bool = true) {
        JunoBroker.postJSON(
            path: "api/broker/settings/live_caption",
            payload: ["enabled": enabled]
        ) { obj in
            let ok = (obj["ok"] as? Bool) ?? false
            if !ok, reportFailures {
                let msg = (obj["error"] as? String) ?? "Could not save live transcription setting"
                JunoSettingsToastCenter.shared.report(msg)
            }
        }
    }

    private var livePreviewResourceMessage: String? {
        if !previewEligibility.isEligible {
            return previewEligibility.unavailableMessage
        }
        if hudLiveTranscriptions {
            return previewEligibility.warningMessage
        }
        return nil
    }

    private func refreshPreviewEligibility() {
        previewEligibility = JunoPreviewEligibility.current
        hudLiveTranscriptions = JunoUserDefaults.hudLiveTranscriptionsEnabled
        if !previewEligibility.isEligible {
            persistLiveCaptionEnabled(false, reportFailures: false)
        }
    }

    private func setLiveTranscriptionPreview(_ enabled: Bool) {
        if enabled && !previewEligibility.isEligible {
            hudLiveTranscriptions = false
            JunoUserDefaults.hudLiveTranscriptionsEnabled = false
            persistLiveCaptionEnabled(false)
            if let message = previewEligibility.unavailableMessage {
                JunoSettingsToastCenter.shared.report(message)
            }
            return
        }

        hudLiveTranscriptions = enabled
        JunoUserDefaults.hudLiveTranscriptionsEnabled = enabled
        persistLiveCaptionEnabled(enabled)

        if enabled {
            if let warning = previewEligibility.warningMessage {
                JunoSettingsToastCenter.shared.report(warning, severity: .info, autoDismissAfter: 8)
            }
        }
    }

    // MARK: Launch at login (SMAppService on macOS 13+; legacy plist cleanup)

    private func currentLaunchAtLoginEnabled() -> Bool {
        // Authoritative source: SMAppService status. Also opportunistically
        // clean up any leftover plist from older builds that used launchctl.
        cleanupLegacyLaunchAgentsIfPresent()
        return SMAppService.mainApp.status == .enabled
    }

    private func legacyPlistURLs() -> [URL] {
        guard let lib = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first else { return [] }
        return [
            lib.appendingPathComponent("LaunchAgents/com.juno.launch.plist"),
            lib.appendingPathComponent("LaunchAgents/com.juno.shell.agent.plist"),
        ]
    }

    private func cleanupLegacyLaunchAgentsIfPresent() {
        let fm = FileManager.default
        for url in legacyPlistURLs() where fm.fileExists(atPath: url.path) {
            _ = launchctlOp("unload", url)
            try? fm.removeItem(at: url)
        }
    }

    private func setLaunchAtLogin(_ enabled: Bool) {
        launchAtLoginError = nil
        cleanupLegacyLaunchAgentsIfPresent()
        let service = SMAppService.mainApp
        do {
            if enabled {
                try service.register()
            } else {
                try service.unregister()
            }
        } catch {
            setLaunchAtLoginToggleWithoutSaving(!enabled)
            launchAtLoginError = enabled
                ? "Could not enable Launch on login: \(error.localizedDescription)"
                : "Could not disable Launch on login: \(error.localizedDescription)"
        }
    }

    private func setLaunchAtLoginToggleWithoutSaving(_ enabled: Bool) {
        suppressLaunchAtLoginOnChange = true
        launchAtLoginEnabled = enabled
        suppressLaunchAtLoginOnChange = false
    }

    private func performStorageAction(_ action: JunoPendingStorageAction) {
        switch action {
        case .clearHistory:
            retention.clearHistory()
        case .deleteRecordings:
            retention.pruneAllAudio()
        case .cleanup:
            retention.runCleanup()
        }
    }

    private func launchctlOp(_ op: String, _ url: URL) -> (ok: Bool, message: String?) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        p.arguments = [op, url.path]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        do {
            try p.run()
            p.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let message = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (p.terminationStatus == 0, message?.isEmpty == false ? message : nil)
        } catch {
            return (false, error.localizedDescription)
        }
    }
}

// MARK: - Retention settings model (broker-backed)

struct JunoRetentionChoice: Identifiable, Hashable {
    let id: String
    let title: String
    let policy: String  // "forever" | "off" | "days"
    let days: Int?

    static let forever = JunoRetentionChoice(id: "forever", title: "Forever", policy: "forever", days: nil)
    static let off = JunoRetentionChoice(id: "off", title: "Off", policy: "off", days: nil)
    static func days(_ d: Int) -> JunoRetentionChoice {
        JunoRetentionChoice(id: "days_\(d)", title: "\(d) days", policy: "days", days: d)
    }

    static let audioCases: [JunoRetentionChoice] = [.forever, .days(7), .days(30), .off]
    static let historyCases: [JunoRetentionChoice] = [.forever, .days(30), .days(90), .off]
}

@MainActor
final class JunoRetentionSettingsModel: ObservableObject {
    @Published var brokerReachable: Bool = true
    @Published var isBusy: Bool = false
    @Published var inlineError: String?

    @Published var audioChoice: JunoRetentionChoice = .days(30)
    @Published var historyChoice: JunoRetentionChoice = .days(90)

    @Published private var storageStats: [String: Any] = [:]
    /// Resolved broker `log_dir` (recordings, history sidecars, etc.); from `GET /api/broker/storage/stats`.
    @Published private(set) var storageLogDir: String?

    var storageSummaryLine: String? {
        guard let ok = storageStats["ok"] as? Bool, ok else { return nil }
        let audioBytes = (storageStats["audio_bytes"] as? Int) ?? 0
        let audioFiles = (storageStats["audio_files"] as? Int) ?? 0
        let historyBytes = (storageStats["history_bytes"] as? Int) ?? 0
        let historyEntries = (storageStats["history_entries"] as? Int) ?? 0
        let mb = { (b: Int) -> String in String(format: "%.1f MB", Double(b) / 1_048_576.0) }
        let total = audioBytes + historyBytes
        return "Using \(mb(total)) — \(audioFiles) recording\(audioFiles == 1 ? "" : "s") and \(historyEntries) history entr\(historyEntries == 1 ? "y" : "ies")."
    }

    func openStorageLogDirInFinder() {
        guard let raw = storageLogDir?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return }
        NSWorkspace.shared.open(URL(fileURLWithPath: raw, isDirectory: true))
    }

    func refresh() {
        inlineError = nil
        brokerReachable = true
        fetchSettings()
        fetchStorageStats()
    }

    func persistAndCleanup() {
        persistSettings { [weak self] ok in
            guard let self else { return }
            if ok { self.runCleanup() }
        }
    }

    func runCleanup() {
        guard !isBusy else { return }
        isBusy = true
        inlineError = nil
        JunoBroker.postJSON(path: "api/broker/retention/run_cleanup", payload: [:]) { [weak self] obj in
            guard let self else { return }
            self.isBusy = false
            let ok = (obj["ok"] as? Bool) ?? false
            if !ok {
                let msg = (obj["error"] as? String) ?? "Cleanup failed"
                self.inlineError = msg
                JunoSettingsToastCenter.shared.report(msg)
            }
            self.fetchStorageStats()
        }
    }

    func clearHistory() {
        guard !isBusy else { return }
        isBusy = true
        inlineError = nil
        JunoBroker.postJSON(path: "api/broker/history/clear_all", payload: [:]) { [weak self] obj in
            guard let self else { return }
            self.isBusy = false
            let ok = (obj["ok"] as? Bool) ?? false
            if !ok {
                let msg = (obj["error"] as? String) ?? "Clear history failed"
                self.inlineError = msg
                JunoSettingsToastCenter.shared.report(msg)
            }
            self.fetchStorageStats()
        }
    }

    func pruneAllAudio() {
        guard !isBusy else { return }
        isBusy = true
        inlineError = nil
        JunoBroker.postJSON(path: "api/broker/storage/audio/prune_all", payload: [:]) { [weak self] obj in
            guard let self else { return }
            self.isBusy = false
            let ok = (obj["ok"] as? Bool) ?? false
            if !ok {
                let msg = (obj["error"] as? String) ?? "Delete recordings failed"
                self.inlineError = msg
                JunoSettingsToastCenter.shared.report(msg)
            }
            self.fetchStorageStats()
        }
    }

    private func fetchSettings() {
        JunoBroker.getJSON(path: "api/broker/settings") { [weak self] obj in
            guard let self else { return }
            let ok = (obj["ok"] as? Bool) ?? false
            if !ok {
                self.brokerReachable = false
                self.inlineError = "Voice engine not connected"
                return
            }
            let settings = (obj["settings"] as? [String: Any]) ?? [:]
            self.audioChoice = self.choiceFromSettings(prefix: "audio_retention", settings: settings, fallback: .days(30), cases: JunoRetentionChoice.audioCases)
            self.historyChoice = self.choiceFromSettings(prefix: "history_retention", settings: settings, fallback: .days(90), cases: JunoRetentionChoice.historyCases)
        }
    }

    private func fetchStorageStats() {
        JunoBroker.getJSON(path: "api/broker/storage/stats") { [weak self] obj in
            guard let self else { return }
            self.storageStats = obj
            if let s = obj["log_dir"] as? String {
                let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
                self.storageLogDir = t.isEmpty ? nil : t
            } else {
                self.storageLogDir = nil
            }
        }
    }

    private func persistSettings(completion: @escaping (Bool) -> Void) {
        guard !isBusy else { completion(false); return }
        isBusy = true
        inlineError = nil
        let payload: [String: Any] = [
            "audio_policy": audioChoice.policy,
            "audio_days": audioChoice.days as Any,
            "history_policy": historyChoice.policy,
            "history_days": historyChoice.days as Any,
        ]
        JunoBroker.postJSON(path: "api/broker/settings/retention", payload: payload) { [weak self] obj in
            guard let self else { return }
            self.isBusy = false
            let ok = (obj["ok"] as? Bool) ?? false
            if !ok {
                let msg = (obj["error"] as? String) ?? "Could not save settings"
                self.inlineError = msg
                JunoSettingsToastCenter.shared.report(msg)
            }
            completion(ok)
        }
    }

    private func choiceFromSettings(
        prefix: String,
        settings: [String: Any],
        fallback: JunoRetentionChoice,
        cases: [JunoRetentionChoice]
    ) -> JunoRetentionChoice {
        let policy = (settings["\(prefix)_policy"] as? String) ?? ""
        let days = settings["\(prefix)_days"] as? Int
        if policy == "forever" { return .forever }
        if policy == "off" { return .off }
        if policy == "days", let d = days {
            if let match = cases.first(where: { $0.policy == "days" && $0.days == d }) { return match }
            return .days(d)
        }
        return fallback
    }
}

@MainActor
final class JunoPrivacySettingsModel: ObservableObject {
    @Published var smartContext = true
    @Published var useSelectedText = true
    @Published var useFocusedText = true
    @Published var useWindowTitle = true
    @Published var useClipboard = false
    @Published var learnFromCorrections = true
    @Published var inlineError: String?
    @Published var brokerReachable: Bool = true

    func refresh() {
        JunoBroker.getJSON(path: "api/broker/privacy/context_settings") { [weak self] obj in
            guard let self else { return }
            let ok = (obj["ok"] as? Bool) ?? false
            guard ok else {
                self.brokerReachable = false
                self.inlineError = "Voice engine not connected"
                return
            }
            self.brokerReachable = true
            let settings = (obj["settings"] as? [String: Any]) ?? [:]
            self.smartContext = (settings["smart_context"] as? Bool) ?? true
            self.useSelectedText = (settings["use_selected_text"] as? Bool) ?? true
            self.useFocusedText = (settings["use_focused_text"] as? Bool) ?? true
            self.useWindowTitle = (settings["use_window_title"] as? Bool) ?? true
            self.useClipboard = (settings["use_clipboard"] as? Bool) ?? false
            self.learnFromCorrections = (settings["learn_from_corrections"] as? Bool) ?? true
            self.inlineError = nil
        }
    }

    func persist() {
        inlineError = nil
        let payload: [String: Any] = [
            "smart_context": smartContext,
            "use_selected_text": useSelectedText,
            "use_focused_text": useFocusedText,
            "use_window_title": useWindowTitle,
            "use_clipboard": useClipboard,
            "learn_from_corrections": learnFromCorrections,
        ]
        JunoBroker.postJSON(path: "api/broker/privacy/context_settings", payload: payload) { [weak self] obj in
            guard let self else { return }
            let ok = (obj["ok"] as? Bool) ?? false
            if !ok {
                let msg = (obj["error"] as? String) ?? "Could not save context settings"
                self.inlineError = msg
                JunoSettingsToastCenter.shared.report(msg)
            }
        }
    }
}

/// Developer-only diagnostics + log tools. Embedded inline inside
/// ``JunoSettingsView`` when ``JunoUserDefaults.developerModeEnabled`` is on
/// — the previous standalone NavigationLink presentation had no back button
/// because the parent's nav toolbar is hidden, so we keep the dev cards in
/// the same scroll surface as the rest of Settings instead.
struct JunoAdvancedSettingsView: View {
    @ObservedObject private var perms = JunoPermissionMonitor.shared
    @ObservedObject private var windowNav = JunoMainWindowNavigator.shared
    @Environment(\.colorScheme) private var scheme

    @State private var saveLogsToFile = JunoUserDefaults.saveLogsToFileEnabled
    @State private var lastBundleURL: URL?
    @State private var bundleStatus: String?
    @State private var pendingClearMemory = false
    @State private var isClearingMemory = false
    @State private var clearMemoryStatus: String?
    @State private var memoryCounts: [String: Int] = [:]
    @State private var isRefreshingMemory = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            advancedCard(title: "Voice engine") {
                    JunoBrokerSetupCard(showDoctorHint: true)
                }

                advancedCard(title: "Onboarding") {
                    Text(
                        "Re-runs the welcome flow and permission cards. To also reset macOS permissions, quit Juno and remove it from System Settings → Privacy & Security, then reopen."
                    )
                    .font(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 10) {
                        Button("Run again…") { JunoOnboardingWindow.show() }
                            .junoPrimaryActionButton()
                        Button("Reset & open onboarding") {
                            JunoUserDefaults.resetOnboardingForRetest()
                            JunoOnboardingWindow.show()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }

                advancedCard(title: "Memory") {
                    Text(
                        "Review and edit learned vocabulary, corrections, replacements, and snippets. Bulk clear also removes session entities. Dictation history and retained recordings stay intact."
                    )
                    .font(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)

                    Text(memorySummaryLine)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)

                    HStack(spacing: 10) {
                        Button("View and edit…") {
                            windowNav.openDictionaryAndMemory(categoryRaw: "vocab")
                        }
                        .junoPrimaryActionButton()

                        Button(isRefreshingMemory ? "Refreshing…" : "Refresh") {
                            refreshMemorySnapshot()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(isRefreshingMemory)

                        Button(isClearingMemory ? "Clearing…" : "Clear learned memory…") {
                            pendingClearMemory = true
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(isClearingMemory)

                        if isClearingMemory {
                            ProgressView()
                                .controlSize(.small)
                                .scaleEffect(0.75)
                        }
                    }

                    if let clearMemoryStatus {
                        Text(clearMemoryStatus)
                            .font(.caption)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                advancedCard(title: "Diagnostics") {
                    Button("Refresh permissions") { perms.refresh() }
                        .buttonStyle(.bordered)
                        .controlSize(.small)

                    Button("Export diagnostics…") {
                        JunoBroker.fetchBinary(path: "api/broker/export/data.zip") { result in
                            guard case .success(let data) = result else { return }
                            let url = FileManager.default.temporaryDirectory
                                .appendingPathComponent("juno-export-\(UUID().uuidString).zip")
                            do {
                                try data.write(to: url, options: .atomic)
                                NSWorkspace.shared.open(url)
                            } catch {
                                NSLog("Juno: failed to write diagnostics export: \(error.localizedDescription)")
                            }
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                advancedCard(title: "Logs") {
                    VStack(alignment: .leading, spacing: 10) {
                        Toggle(isOn: $saveLogsToFile) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Save app logs to file")
                                    .font(.system(.callout, design: .rounded).weight(.semibold))
                                    .foregroundStyle(JunoTheme.primaryText(scheme))
                                Text("Saves Juno's logs to disk so you can attach them to support requests. Takes effect on next launch.")
                                    .font(.caption)
                                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                        .toggleStyle(.switch)
                        .onChange(of: saveLogsToFile) { newValue in
                            JunoUserDefaults.saveLogsToFileEnabled = newValue
                        }

                        Divider().opacity(0.35)

                        Text(JunoSupportBundle.logDirectoryDisplayPath)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)

                        HStack(spacing: 8) {
                            Button("Show logs in Finder") {
                                JunoSupportBundle.revealLogDirectory()
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)

                            Button("Generate support bundle…") {
                                if let url = JunoSupportBundle.generateAndReveal() {
                                    lastBundleURL = url
                                    bundleStatus = "Saved \(url.lastPathComponent) — Finder window is open."
                                } else {
                                    bundleStatus = "Could not write support bundle. Check ~/Library/Logs/Juno permissions."
                                }
                            }
                            .junoPrimaryActionButton()
                        }

                        if let bundleStatus {
                            Text(bundleStatus)
                                .font(.caption)
                                .foregroundStyle(JunoTheme.secondaryText(scheme))
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }

        }
        .onAppear {
            perms.refresh()
            refreshMemorySnapshot()
        }
        .confirmationDialog(
            "Clear learned memory?",
            isPresented: $pendingClearMemory,
            titleVisibility: .visible
        ) {
            Button("Clear Learned Memory", role: .destructive) { clearLearnedMemory() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This keeps dictation history and recordings, but removes learned transcription memory. Use it when Juno keeps biasing toward the wrong words.")
        }
    }

    private func clearLearnedMemory() {
        guard !isClearingMemory else { return }
        isClearingMemory = true
        clearMemoryStatus = nil
        JunoBroker.postJSON(path: "api/broker/memory/clear_all", payload: [:]) { obj in
            let ok = (obj["ok"] as? Bool) ?? false
            isClearingMemory = false
            if ok {
                let removedCounts = (obj["removed"] as? [String: Any]) ?? [:]
                let removed = removedCounts.values.reduce(0) { total, value in
                    total + ((value as? Int) ?? 0)
                }
                clearMemoryStatus = removed > 0
                    ? "Cleared \(removed) learned memory item\(removed == 1 ? "" : "s"). History was not changed."
                    : "Learned memory was already clear. History was not changed."
                refreshMemorySnapshot()
            } else {
                let msg = (obj["error"] as? String) ?? "Could not clear learned memory"
                clearMemoryStatus = msg
                JunoSettingsToastCenter.shared.report(msg)
            }
        }
    }

    private var memorySummaryLine: String {
        guard !memoryCounts.isEmpty else { return "Memory: loading…" }
        let vocab = memoryCounts["lexicon"] ?? 0
        let corrections = memoryCounts["corrections"] ?? 0
        let replacements = memoryCounts["replacements"] ?? 0
        let snippets = memoryCounts["snippets"] ?? 0
        let entities = memoryCounts["session_entities"] ?? 0
        return "Memory: \(vocab) vocab, \(corrections) corrections, \(replacements) replacements, \(snippets) snippets, \(entities) session entities"
    }

    private func refreshMemorySnapshot() {
        guard !isRefreshingMemory else { return }
        isRefreshingMemory = true
        JunoBroker.getJSON(path: "api/broker/memory/snapshot") { obj in
            isRefreshingMemory = false
            let ok = (obj["ok"] as? Bool) ?? false
            if ok, let counts = obj["counts"] as? [String: Any] {
                memoryCounts = counts.reduce(into: [String: Int]()) { partial, pair in
                    partial[pair.key] = (pair.value as? Int) ?? 0
                }
            }
        }
    }

    private func advancedCard(title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            JunoSectionLabel(text: title)
            content()
        }
        .padding(16)
        .premiumCard()
    }
}

// MARK: - Settings → Actions pointer

/// Slim card on Settings that points users to the Actions sidebar page.
/// Replaces the old in-Settings VoiceActionsBanner so all per-action
/// state, permissions, and copy live in one place.
private struct JunoSettingsActionsPointer: View {
    @StateObject private var perms = JunoActionPermissionStore.shared
    @ObservedObject private var windowNav = JunoMainWindowNavigator.shared
    @AppStorage(JunoUserDefaults.actionsEnabledKey) private var actionsEnabled: Bool = false
    @Environment(\.colorScheme) private var scheme

    private var summary: String {
        if !actionsEnabled { return "Off" }
        let granted = JunoActionCatalogAll.filter { perms.status(for: $0.permission).isGranted }.count
        return "\(granted) of \(JunoActionCatalogAll.count) ready"
    }

    var body: some View {
        Button {
            windowNav.section = .actions
        } label: {
            HStack(spacing: 12) {
                Image(systemName: "bolt.badge.checkmark")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(JunoDesignTokens.accent)
                    .frame(width: 22)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Voice Actions")
                        .font(.system(.subheadline, design: .rounded).weight(.semibold))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text("Notes, reminders, and alarms by voice — \(summary).")
                        .font(.system(.footnote, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .premiumCard()
        }
        .buttonStyle(.plain)
        .focusEffectDisabled()
        .onAppear { perms.refreshAll() }
    }
}


// MARK: - Update available banner
//
// Promoted, accent-tinted strip that appears in Settings → App when an
// update is staged. Replaces the previous "Install…" button buried in a
// settingsRow trailing slot: update presence
// wasn't surfaced anywhere the user would notice. The banner is the
// place users land when they click the Updates dot from the menu bar.
struct JunoUpdateAvailableBanner: View {
    let version: String?
    let onInstall: () -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(alignment: .center, spacing: JunoUI.Spacing.m) {
            ZStack {
                Circle()
                    .fill(JunoDesignTokens.accent.opacity(scheme == .dark ? 0.16 : 0.10))
                    .frame(width: 32, height: 32)
                Image(systemName: "arrow.down.circle.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(JunoDesignTokens.accent)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(version.map { "Update \($0) is ready" } ?? "An update is ready")
                    .junoType(.bodyEmphasis)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("Install on next quit, or restart now to apply.")
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }
            Spacer(minLength: 0)
            Button("Install…", action: onInstall)
                .junoPrimaryActionButton()
        }
        .padding(.horizontal, JunoUI.Spacing.m)
        .padding(.vertical, JunoUI.Spacing.s + 2)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(JunoDesignTokens.accent.opacity(scheme == .dark ? 0.08 : 0.05))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(JunoDesignTokens.accent.opacity(scheme == .dark ? 0.32 : 0.20), lineWidth: 0.8)
        )
    }
}
