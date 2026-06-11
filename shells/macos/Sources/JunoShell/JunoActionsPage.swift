// JunoActionsPage.swift
//
// The "Actions" sidebar destination. The page has two surfaces:
// a single main card that adapts to the feature's state, plus an
// Advanced disclosure. Each row owns its own permission affordance
// and its own example utterance, so no section stacks duplicate
// "this is allowed" / "try saying" copy. The editing/transform
// voice commands live on their own "Voice Commands" page — they
// don't need any of these permissions.
//
// State of the main card:
//   - actionsEnabled == false                  → empty-state hero
//   - actionsEnabled == true, all granted      → ready rows + inline examples
//   - actionsEnabled == true, some missing     → mixed rows with Allow CTAs

import AppKit
import SwiftUI

struct JunoActionsPage: View {
    @StateObject private var perms = JunoActionPermissionStore.shared
    @AppStorage(JunoUserDefaults.actionsEnabledKey) private var actionsEnabled: Bool = false
    @AppStorage(JunoUserDefaults.actionsNotesSignatureEnabledKey) private var notesSignature: Bool = true
    @Environment(\.colorScheme) private var scheme

    @State private var notesHelpVisible: Bool = false
    @State private var advancedExpanded: Bool = false
    /// Tracks how many times the user has clicked Allow on Notes without
    /// the status leaving `.notDetermined`. After two no-op clicks we
    /// auto-reveal the help block since the consent dialog clearly isn't
    /// reaching them.
    @State private var notesAllowAttempts: Int = 0

    private var allGranted: Bool {
        JunoActionCatalogAll.allSatisfy { perms.status(for: $0.permission).isGranted }
    }

    private var grantedDescriptors: [JunoActionDescriptor] {
        JunoActionCatalogAll.filter { perms.status(for: $0.permission).isGranted }
    }

    private var missingDescriptors: [JunoActionDescriptor] {
        JunoActionCatalogAll.filter { !perms.status(for: $0.permission).isGranted }
    }

    /// One representative example per action — the first entry from the
    /// shared catalog. We render this as the "Try saying" line inside
    /// each row, replacing the old standalone "Try saying" card.
    private func primaryExample(for d: JunoActionDescriptor) -> String {
        d.examples.first ?? ""
    }

    /// Where each action lands once Juno has saved it. Kept to a noun
    /// phrase, not a sentence — Apple-style "iCloud · Photos"-shape.
    private func destinationLabel(for d: JunoActionDescriptor) -> String {
        switch d.kind {
        case .reminder: return "Reminders"
        case .note:     return "Notes · Juno folder"
        case .alarm:    return "Alarm"
        }
    }

    var body: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 18) {
                header
                mainCard
                advancedSection
            }
            .junoDetailPagePadding()
        }
        .onAppear { perms.beginObserving() }
        .onDisappear { perms.endObserving() }
    }

    // MARK: - Header

    private var header: some View {
        JunoPageHeader(
            eyebrow: "Actions",
            title: "Voice Actions",
            subtitle: nil,
            trailing: {
                Button {
                    perms.refreshAll(forceNotesProbe: true)
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                        .font(.system(size: 12, weight: .semibold))
                }
                .junoSecondaryActionButton()
                .help("Refresh macOS permission status")
            }
        )
    }

    // MARK: - Main card (state-driven)

    @ViewBuilder
    private var mainCard: some View {
        if !actionsEnabled {
            emptyStateHero
        } else {
            actionsReadyCard
        }
    }

    // MARK: - Empty state (Voice Actions off)
    //
    // Same shape as ``actionsReadyCard``: title + caption + master
    // toggle on top, then the three action rows. Flipping the toggle
    // is the only visual change between off and on — no separate hero
    // motif, no big marketing button at the bottom. The rows show the
    // example utterance + destination so the user can read what they're
    // turning on without it feeling like a decorative preview.

    private var emptyStateHero: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Voice Actions")
                        .junoType(.bodyEmphasis)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text("Off — turn on to save reminders, notes, and alarms by voice.")
                        .junoType(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                Toggle("", isOn: $actionsEnabled)
                    .toggleStyle(.switch)
                    .labelsHidden()
            }

            VStack(spacing: 0) {
                ForEach(Array(JunoActionCatalogAll.enumerated()), id: \.element.id) { index, descriptor in
                    emptyStateExampleRow(descriptor)
                    if index < JunoActionCatalogAll.count - 1 {
                        Divider().opacity(0.20).padding(.leading, 40)
                    }
                }
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .premiumCard()
    }

    /// Same vertical structure as a granted ``actionRow``: header on top,
    /// example utterance + destination indented below. Keeping the
    /// layouts identical means the toggle flipping doesn't reflow the
    /// page — just the trailing affordance changes (nothing → Allow →
    /// Allowed).
    private func emptyStateExampleRow(_ d: JunoActionDescriptor) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 12) {
                JunoActionNativeIconTile(kind: d.kind, tileSize: 28, iconSize: 24, fallbackTint: d.accent)
                Text(d.pluralName)
                    .font(.system(.footnote, design: .rounded).weight(.semibold))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Spacer(minLength: 0)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("\u{201C}\(primaryExampleTrimmed(d: d))\u{201D}")
                    .font(.system(.subheadline, design: .rounded))
                    .italic()
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
                Text(destinationLabel(for: d))
                    .font(.system(size: 10.5, weight: .semibold, design: .monospaced))
                    .tracking(0.6)
                    .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.72))
            }
            .padding(.leading, 40)
        }
        .padding(.vertical, 10)
    }

    /// Convenience: the catalog's example text shortened by trimming the
    /// trailing ellipsis when present, so the empty-state lines stay tidy.
    private func primaryExampleTrimmed(d: JunoActionDescriptor) -> String {
        let raw = primaryExample(for: d)
        return raw.replacingOccurrences(of: "\u{2026}", with: "")
    }

    // MARK: - Actions ready card (states B and C unified)
    //
    // The page's main control surface when Voice Actions is on. Each row
    // carries its action's name, status, example utterance (when granted),
    // and destination — no separate "try saying" section, no double "X of
    // Y ready" headlines.

    private var actionsReadyCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Compact master row — toggle + concise state caption.
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Voice Actions")
                        .junoType(.bodyEmphasis)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text(masterCaption)
                        .junoType(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
                Spacer(minLength: 0)
                Toggle("", isOn: $actionsEnabled)
                    .toggleStyle(.switch)
                    .labelsHidden()
            }

            VStack(spacing: 0) {
                ForEach(Array(JunoActionCatalogAll.enumerated()), id: \.element.id) { index, descriptor in
                    actionRow(descriptor)
                    if index < JunoActionCatalogAll.count - 1 {
                        Divider().opacity(0.20).padding(.leading, 40)
                    }
                }
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .premiumCard()
    }

    /// One-line caption under the master toggle. Apple-style: states
    /// fact, never marketing prose. Hidden entirely when everything is
    /// allowed (the per-row state speaks for itself).
    private var masterCaption: String {
        if allGranted { return "All set." }
        let missing = missingDescriptors.count
        if missing == 1 { return "One left to allow." }
        return "\(missing) left to allow."
    }

    private func actionRow(_ d: JunoActionDescriptor) -> some View {
        let status = perms.status(for: d.permission)
        let isBusy = perms.isRequesting(d.permission)
        let granted = status.isGranted
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 12) {
                JunoActionNativeIconTile(kind: d.kind, tileSize: 28, iconSize: 24, fallbackTint: d.accent)
                Text(d.pluralName)
                    .font(.system(.footnote, design: .rounded).weight(.semibold))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Spacer(minLength: 0)
                rightTrailing(d: d, status: status, isBusy: isBusy)
            }

            if granted {
                // Granted: example utterance + destination, tidy two-line block.
                VStack(alignment: .leading, spacing: 2) {
                    Text("\u{201C}\(primaryExample(for: d))\u{201D}")
                        .font(.system(.subheadline, design: .rounded))
                        .italic()
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                    Text(destinationLabel(for: d))
                        .font(.system(size: 10.5, weight: .semibold, design: .monospaced))
                        .tracking(0.6)
                        .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.72))
                }
                .padding(.leading, 40)
            } else if status.needsUserAction {
                // Pre-allow: short teaser of where this lands.
                Text(setupTeaser(for: d))
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, 40)

                if d.permission == .notesAutomation && notesHelpVisible {
                    notesHelpBlock(d)
                        .padding(.leading, 40)
                }
            }
        }
        .padding(.vertical, 10)
    }

    /// Right-side trailing element for each row: granted → small Allowed
    /// pill; pre-allow → Allow primary button; denied → Open Settings
    /// secondary; busy → spinner.
    @ViewBuilder
    private func rightTrailing(d: JunoActionDescriptor, status: JunoActionPermissionStatus, isBusy: Bool) -> some View {
        if isBusy {
            ProgressView().controlSize(.small)
        } else if status.isGranted {
            HStack(spacing: 5) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 10, weight: .semibold))
                Text("Allowed")
                    .font(.system(.caption, design: .rounded).weight(.semibold))
            }
            .foregroundStyle(JunoDesignTokens.meadow)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Capsule().fill(JunoDesignTokens.meadow.opacity(0.12)))
        } else if status == .notDetermined {
            HStack(spacing: 6) {
                Button("Allow") { handleAllow(d) }
                    .buttonStyle(JunoPrimaryActionButtonStyle())
                    .controlSize(.small)
                    .junoNoFocusRing()
                if d.permission == .notesAutomation {
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) { notesHelpVisible.toggle() }
                    } label: {
                        Image(systemName: notesHelpVisible ? "questionmark.circle.fill" : "questionmark.circle")
                            .font(.system(size: 13, weight: .semibold))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .help("What if the popup doesn't appear?")
                }
            }
        } else {
            Button("Open Settings") { handleOpenSettings(d) }
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
    }

    /// One sentence per action explaining the destination, used in setup
    /// mode where the example utterance is held back behind the Allow.
    private func setupTeaser(for d: JunoActionDescriptor) -> String {
        switch d.kind {
        case .reminder: return "Adds to-dos in Reminders."
        case .note:     return "Saves notes to a folder called Juno in Notes."
        case .alarm:    return "Creates an alarm that rings on time."
        }
    }

    private func notesHelpBlock(_ d: JunoActionDescriptor) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Click Allow first. macOS may briefly open Notes so it can show the real permission prompt. If the prompt still doesn't appear, open Automation settings and turn on Notes for Juno.")
                .font(.system(.caption, design: .rounded))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .fixedSize(horizontal: false, vertical: true)
            Button("Open Automation Settings") { handleOpenSettings(d) }
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
    }

    // MARK: - Advanced (moved out of the main card)

    private var advancedSection: some View {
        DisclosureGroup(isExpanded: $advancedExpanded) {
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Add a footer to saved notes")
                        .font(.system(.footnote, design: .rounded).weight(.medium))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text("Appends \u{201C}Captured with Juno · {time}\u{201D} at the bottom of every note Juno saves.")
                        .font(.system(.caption, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                Toggle("", isOn: $notesSignature)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .labelsHidden()
            }
            .padding(.top, 10)
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                VStack(alignment: .leading, spacing: 2) {
                    Text("Advanced")
                        .junoType(.bodyEmphasis)
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text("Fine-grained settings for saved actions.")
                        .junoType(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .premiumCard()
    }


    // MARK: - Permission button handlers

    private func handleAllow(_ descriptor: JunoActionDescriptor) {
        JunoWindowActivation.activateApp()

        let onResolved: (JunoActionPermissionStatus) -> Void = { status in
            // Turn the feature on as soon as any requested action is
            // actually granted. Notes and reminders are independent; the
            // user should not have to finish every permission just to use
            // the one they enabled.
            if status == .granted && !actionsEnabled {
                actionsEnabled = true
            }
            // Notes-specific UX nicety: if the consent dialog clearly
            // didn't show (status didn't move), auto-open the help
            // block on the second attempt so the user sees the System
            // Settings escape hatch without having to discover it.
            if descriptor.permission == .notesAutomation {
                if status == .notDetermined {
                    notesAllowAttempts += 1
                    withAnimation(.easeInOut(duration: 0.2)) {
                        notesHelpVisible = true
                    }
                    if notesAllowAttempts >= 2 {
                        perms.openAutomationSettings()
                    }
                } else {
                    notesAllowAttempts = 0
                }
            }
        }

        switch descriptor.permission {
        case .reminders:
            perms.requestReminders(onResolved)
        case .calendarEvents:
            perms.requestCalendarEvents(onResolved)
        case .notesAutomation:
            perms.requestNotesAutomation(onResolved)
        }
    }

    private func handleOpenSettings(_ descriptor: JunoActionDescriptor) {
        // Use the multi-fallback opener — the older single-URL approach
        // silently no-op'd on macOS Sonoma/Sequoia when the deep-link
        // form drifted between releases.
        switch descriptor.permission {
        case .notesAutomation:
            perms.openAutomationSettings()
        case .reminders:
            perms.openRemindersSettings()
        case .calendarEvents:
            perms.openCalendarSettings()
        }
    }
}

// MARK: - Tiny flow layout

/// Drop-in flow layout for chip rows. Stays SwiftUI-native (no AppKit
/// hosting) so the page reads on small windows the same as wide ones.
struct JunoFlowLayout: Layout {
    var spacing: CGFloat = 6
    var runSpacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        var totalWidth: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth, x > 0 {
                x = 0
                y += rowHeight + runSpacing
                rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
            totalWidth = max(totalWidth, x)
        }
        let contentHeight = y + rowHeight
        return CGSize(width: maxWidth.isFinite ? maxWidth : totalWidth, height: contentHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let maxWidth = bounds.width
        var x: CGFloat = bounds.minX
        var y: CGFloat = bounds.minY
        var rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > bounds.minX + maxWidth, x > bounds.minX {
                x = bounds.minX
                y += rowHeight + runSpacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(width: size.width, height: size.height))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}
