import AppKit
import AVFoundation
import SwiftUI

// MARK: - Defaults

private enum JunoOnboardingDefaults {
    static var isCompleted: Bool {
        get { JunoUserDefaults.onboardingCompleted }
        set { JunoUserDefaults.onboardingCompleted = newValue }
    }
}

// MARK: - Permission helpers

enum JunoPermissions {
    static var micStatus: AVAuthorizationStatus { AVCaptureDevice.authorizationStatus(for: .audio) }
    static func requestMic(completion: @escaping (Bool) -> Void) {
        AVCaptureDevice.requestAccess(for: .audio) { granted in DispatchQueue.main.async { completion(granted) } }
    }
    static func openMicSettings()              { JunoSystemSettingsLinks.openMicrophonePrivacy() }
    static func openAXSettings()              { JunoSystemSettingsLinks.openAccessibilityPrivacy() }
    static func openScreenRecordingSettings() { JunoScreenContextAccess.openSystemSettings() }
}

private struct OnboardingSurface<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding(18)
            .premiumCard()
    }
}

private struct OnboardingStepCallout: View {
    let title: String
    let bodyText: String
    let icon: String

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(JunoDesignTokens.accent)
                .frame(width: 24, height: 24)
                .background(
                    Circle()
                        .fill(JunoDesignTokens.accent.opacity(0.10))
                )
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(.subheadline, design: .rounded).weight(.semibold))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(bodyText)
                    .font(.subheadline)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme).opacity(0.42), lineWidth: 0.5)
        )
    }
}

// MARK: - Step 1: Welcome / Intro

private struct OnboardingIntroStep: View {
    @Binding var preferredName: String
    let onSubmit: () -> Void
    @State private var markVisible = false
    @State private var bodyVisible = false
    @FocusState private var nameFocused: Bool
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .center, spacing: 28) {
            Spacer(minLength: 0)
            heroMark
                .opacity(markVisible ? 1 : 0)
                .scaleEffect(markVisible ? 1 : 0.86)
                .animation(.spring(response: 0.52, dampingFraction: 0.74), value: markVisible)

            VStack(spacing: 10) {
                Text("Speak naturally.")
                    .font(.system(size: 38, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .tracking(-0.4)
                Text("Juno writes where you already are.")
                    .font(.system(size: 20, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .multilineTextAlignment(.center)
            }
            .opacity(bodyVisible ? 1 : 0)
            .offset(y: bodyVisible ? 0 : 8)
            .animation(.easeOut(duration: 0.38).delay(0.12), value: bodyVisible)

            nameField
                .opacity(bodyVisible ? 1 : 0)
                .offset(y: bodyVisible ? 0 : 10)
                .animation(.easeOut(duration: 0.4).delay(0.22), value: bodyVisible)
            if let requirementWarning = JunoSystemRequirements.current.onboardingWarningMessage {
                requirementsBanner(requirementWarning)
                    .opacity(bodyVisible ? 1 : 0)
                    .offset(y: bodyVisible ? 0 : 10)
                    .animation(.easeOut(duration: 0.4).delay(0.30), value: bodyVisible)
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            markVisible = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
                bodyVisible = true
                // Auto-focus so the user can type immediately, then keep
                // pressing Return to walk through the rest of onboarding.
                nameFocused = true
            }
        }
    }

    private var heroMark: some View {
        ZStack {
            // Concentric hairline rings — quiet brand resonance, monochrome.
            Circle()
                .strokeBorder(JunoTheme.border(scheme).opacity(0.18), lineWidth: 0.5)
                .frame(width: 220, height: 220)
            Circle()
                .strokeBorder(JunoTheme.border(scheme).opacity(0.30), lineWidth: 0.5)
                .frame(width: 168, height: 168)
            Circle()
                .fill(JunoDesignTokens.iconBg)
                .frame(width: 132, height: 132)
            JunoCommaMark(color: .white, scale: 1.02)
                .frame(width: 66, height: 90)
        }
        .shadow(color: Color.black.opacity(scheme == .dark ? 0.45 : 0.12), radius: 22, y: 10)
        .accessibilityHidden(true)
    }

    /// Non-blocking memory-floor warning. Juno still runs below the recommended
    /// RAM; this just sets expectations during onboarding (see
    /// ``JunoSystemRequirements``). The macOS-version floor is a hard block
    /// handled at launch, so it never reaches this screen.
    private func requirementsBanner(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(JunoDesignTokens.warning)
            Text(message)
                .font(.system(size: 12.5, weight: .regular, design: .rounded))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .fixedSize(horizontal: false, vertical: true)
                .multilineTextAlignment(.leading)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .frame(maxWidth: 420, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(JunoDesignTokens.warning.opacity(0.10))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(JunoDesignTokens.warning.opacity(0.32), lineWidth: 0.6)
        )
    }

    private var nameField: some View {
        VStack(spacing: 8) {
            Text("WHAT SHOULD WE CALL YOU?")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .tracking(1.2)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
            TextField("Your name (optional)", text: $preferredName)
                .textFieldStyle(.plain)
                .focusEffectDisabled()
                .multilineTextAlignment(.center)
                .font(.system(size: 18, weight: .regular, design: .rounded))
                .focused($nameFocused)
                .onSubmit {
                    nameFocused = false
                    onSubmit()
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .frame(width: 360)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(JunoTheme.cardBackground(scheme))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(
                            nameFocused ? JunoDesignTokens.accent.opacity(0.55) : JunoTheme.border(scheme).opacity(0.45),
                            lineWidth: nameFocused ? 1.2 : 0.6
                        )
                )
                .animation(.easeOut(duration: 0.18), value: nameFocused)
        }
    }
}

// MARK: - Step 2: Permissions

private struct OnboardingPermissionsStep: View {
    @ObservedObject private var perms = JunoPermissionMonitor.shared
    let onRefresh: () -> Void
    @Environment(\.colorScheme) private var scheme

    private var allCore: Bool { perms.micStatus == .authorized && perms.axGranted }
    private var screenContextReady: Bool { perms.screenContextEnabled && perms.screenRecordingGranted }

    var body: some View {
        VStack(alignment: .center, spacing: 16) {
            permissionGlyphRow

            VStack(spacing: 6) {
                Text(allCore ? "You're all set" : "Allow access")
                    .font(.system(size: 28, weight: .semibold, design: .rounded))
                    .tracking(-0.3)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(headerSubtitle)
                    .font(.system(size: 15, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: 460)

            VStack(spacing: 8) {
                permCard(
                    icon: "mic.fill",
                    label: "Microphone",
                    detail: "Lets Juno hear you while dictation is active.",
                    ok: perms.micStatus == .authorized,
                    required: true,
                    primaryLabel: micPrimaryTitle,
                    primary: {
                        switch perms.micStatus {
                        case .notDetermined: perms.requestMic { _ in onRefresh() }
                        default: perms.openMicSettings()
                        }
                    },
                    showSecondarySettings: micShowSecondarySettings,
                    settings: { JunoPermissions.openMicSettings() }
                )
                permCard(
                    icon: "hand.raised.fill",
                    label: "Accessibility",
                    detail: "Lets Juno place text into the app you’re already using.",
                    ok: perms.axGranted,
                    required: true,
                    primaryLabel: "Open Accessibility",
                    primary: {
                        openAccessibilitySettings()
                    },
                    showSecondarySettings: false,
                    settings: { JunoPermissions.openAXSettings() }
                )
                permCard(
                    icon: "viewfinder",
                    label: "Visible screen text",
                    detail: "Adds Juno to macOS Screen Recording so visible names and code terms can be read locally while you dictate.",
                    ok: screenContextReady,
                    required: false,
                    primaryLabel: screenRecordingPrimaryTitle,
                    primary: {
                        perms.requestScreenRecording { _ in
                            onRefresh()
                        }
                    },
                    showSecondarySettings: false,
                    settings: { JunoPermissions.openScreenRecordingSettings() },
                    readyText: "Ready for local screen text."
                )
            }
            .frame(maxWidth: 540)

            Spacer(minLength: 0)

            if !allCore {
                HStack(spacing: 10) {
                    Image(systemName: "info.circle")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                    Text("Don’t see Juno after granting Accessibility? Quit Juno, reopen it, and tap Check permissions.")
                        .font(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer()
                    Button("Check permissions") { perms.refresh(); onRefresh() }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                }
                .padding(.horizontal, 14).padding(.vertical, 10)
                .frame(maxWidth: 540)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(JunoTheme.elevatedCard(scheme).opacity(0.6))
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .onAppear {
            perms.refresh()
        }
        // Polling is owned by JunoPermissionMonitor (every 2 s + app-active /
        // workspace-activate observers). Don't add a second loop here.
    }

    private var headerSubtitle: String {
        if allCore {
            return screenContextReady
                ? "Microphone, Accessibility, and visible screen text are ready."
                : "Microphone and Accessibility are ready. Visible screen text is optional."
        }
        return "Microphone hears you. Accessibility writes where you’re typing. Visible screen text helps spell on-screen terms."
    }

    /// Three monochrome glyphs as the step's hero. Each glyph gains a hairline
    /// ring once its permission is granted — quiet, ink-only feedback that
    /// avoids the cliché green checkmark while still rewarding the action.
    private var permissionGlyphRow: some View {
        HStack(spacing: 24) {
            permissionGlyph(symbol: "mic", granted: perms.micStatus == .authorized, dim: false)
            permissionGlyph(symbol: "hand.raised", granted: perms.axGranted, dim: false)
            permissionGlyph(symbol: "viewfinder", granted: screenContextReady, dim: true)
        }
        .accessibilityHidden(true)
    }

    /// `dim: true` is used for optional permissions so that an un-granted
    /// optional glyph reads quieter than an un-granted required one — without
    /// adding any color, only weight and opacity.
    private func permissionGlyph(symbol: String, granted: Bool, dim: Bool) -> some View {
        let baseColor: Color = granted
            ? JunoTheme.primaryText(scheme)
            : JunoTheme.secondaryText(scheme).opacity(dim ? 0.55 : 0.85)
        return ZStack {
            Circle()
                .strokeBorder(
                    granted
                        ? JunoTheme.primaryText(scheme).opacity(0.85)
                        : JunoTheme.border(scheme).opacity(0.0),
                    lineWidth: 0.8
                )
                .frame(width: 76, height: 76)
                .animation(.easeOut(duration: 0.32), value: granted)
            Image(systemName: granted ? "\(symbol).fill" : symbol)
                .font(.system(size: 31, weight: granted ? .semibold : .regular))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(baseColor)
                .animation(.easeOut(duration: 0.22), value: granted)
        }
        .frame(width: 76, height: 76)
    }

    private func openAccessibilitySettings() {
        perms.openAccessibilitySettings()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            perms.refresh()
            onRefresh()
        }
    }

    private var micPrimaryTitle: String {
        switch perms.micStatus {
        case .notDetermined: return "Allow microphone"
        case .denied, .restricted: return "Open Microphone privacy"
        case .authorized: return "Granted"
        @unknown default: return "Open Microphone privacy"
        }
    }

    /// Never duplicate the primary action: for denied mic the primary already opens System Settings.
    private var micShowSecondarySettings: Bool { false }

    private var screenRecordingPrimaryTitle: String {
        if screenContextReady { return "Granted" }
        return "Open Screen Recording"
    }

    private func permCard(icon: String, label: String, detail: String,
                          ok: Bool, required: Bool,
                          primaryLabel: String,
                          primary: @escaping () -> Void,
                          showSecondarySettings: Bool,
                          settings: @escaping () -> Void,
                          readyText: String? = nil) -> some View {
        HStack(alignment: .center, spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(ok ? JunoDesignTokens.meadow.opacity(0.10) : JunoTheme.elevatedCard(scheme))
                    .frame(width: 44, height: 44)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(
                                ok ? JunoDesignTokens.meadow.opacity(0.38) : JunoDesignTokens.accent.opacity(0.22),
                                lineWidth: ok ? 1 : 0.75
                            )
                    )
                Image(systemName: ok ? "checkmark" : icon)
                    .font(.system(size: ok ? 15 : 16, weight: ok ? .bold : .semibold))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(ok ? JunoDesignTokens.meadow : JunoDesignTokens.accent)
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(label)
                        .font(.system(.callout, design: .rounded).weight(.semibold))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    // No badges — every access on this screen is something
                    // Juno expects. The card hierarchy (icon + label + detail)
                    // is enough; "REQUIRED"/"OPTIONAL" framing made the live-
                    // captions row look skippable.
                }
                Text(detail)
                    .font(.subheadline)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                if ok {
                    Text(readyText ?? (required ? "Ready for dictation." : "Ready."))
                        .font(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
            }
            .layoutPriority(1)
            Spacer(minLength: 8)
            if !ok {
                VStack(alignment: .trailing, spacing: 6) {
                    Button(primaryLabel) { primary() }
                        .controlSize(.regular)
                        .buttonStyle(JunoPrimaryActionButtonStyle())
                        .junoNoFocusRing()
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                    if showSecondarySettings {
                        Button("Open System Settings") { settings() }
                            .buttonStyle(.bordered)
                            .controlSize(.regular)
                    }
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, ok ? 11 : 14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(ok ? JunoDesignTokens.meadow.opacity(0.045) : JunoTheme.cardBackground(scheme))
                .shadow(color: Color.black.opacity(scheme == .dark ? 0.22 : 0.05), radius: ok ? 4 : 10, y: ok ? 1 : 4)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(
                    ok ? JunoDesignTokens.meadow.opacity(0.22) : JunoTheme.border(scheme).opacity(0.55),
                    lineWidth: ok ? 0.5 : 0.5
                )
        )
        .animation(JunoDesignTokens.pillSpring, value: ok)
    }

    private var requiredBadge: some View {
        Text("REQUIRED")
            .font(.system(size: 8, weight: .bold, design: .monospaced))
            .tracking(0.8)
            .foregroundStyle(JunoDesignTokens.accent)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(Capsule().fill(JunoDesignTokens.accent.opacity(0.12)))
    }

    private var optionalBadge: some View {
        Text("OPTIONAL")
            .font(.system(size: 8, weight: .semibold, design: .monospaced))
            .tracking(0.8)
            .foregroundStyle(JunoTheme.secondaryText(scheme))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(Capsule().fill(JunoDesignTokens.muted.opacity(0.12)))
    }
}

// MARK: - Step 3: Activation (shortcut)

private struct OnboardingActivationStep: View {
    @State private var selected: JunoShortcutPreference = JunoShortcutPreference.stored
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(alignment: .center, spacing: 22) {
            heroKeycap
                .padding(.top, 4)

            VStack(spacing: 6) {
                Text("Pick your shortcut")
                    .font(.system(size: 30, weight: .semibold, design: .rounded))
                    .tracking(-0.3)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("Press to start dictating. Press again to land the line.")
                    .font(.system(size: 15, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }
            .frame(maxWidth: 460)

            VStack(spacing: 8) {
                ForEach(JunoShortcutPreference.allCases, id: \.self) { pref in
                    shortcutRow(pref: pref)
                }
            }
            .frame(maxWidth: 540)

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    /// A single oversized keycap — the step's silent protagonist. The label
    /// cross-fades when the user picks a different shortcut. Pure ink + paper.
    private var heroKeycap: some View {
        let labels = keyTiles()
        return HStack(spacing: 10) {
            ForEach(Array(labels.enumerated()), id: \.offset) { idx, label in
                if idx > 0 {
                    Text("+")
                        .font(.system(size: 28, weight: .light, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.55))
                }
                bigKeycap(label)
            }
        }
        .frame(height: 152)
        .accessibilityLabel(Text(selected.displayName))
    }

    private func bigKeycap(_ label: String) -> some View {
        let isWide = label == "Space"
        let isLetter = label.count > 1 && !isWide
        return ZStack {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(JunoTheme.cardBackground(scheme))
                .shadow(color: Color.black.opacity(scheme == .dark ? 0.32 : 0.10), radius: 14, y: 6)
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme).opacity(0.55), lineWidth: 0.8)
            // Quiet inner highlight — keycap topface, not an accent ring.
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.white.opacity(scheme == .dark ? 0.04 : 0.50), lineWidth: 0.5)
                .padding(2)
            Text(label)
                .font(.system(
                    size: isWide ? 26 : (isLetter ? 30 : 64),
                    weight: .medium,
                    design: .rounded
                ))
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .id(label) // forces cross-fade when shortcut changes
                .transition(.opacity.combined(with: .scale(scale: 0.96)))
        }
        .frame(width: isWide ? 220 : 132, height: 132)
        .animation(.easeOut(duration: 0.28), value: label)
    }

    private func keyTiles() -> [String] {
        switch selected {
        case .fn: return ["fn"]
        case .rightCommand: return ["⌘"]
        case .rightOption: return ["⌥"]
        case .optionSpace: return ["⌥", "Space"]
        case .controlSpace: return ["⌃", "Space"]
        }
    }

    private func inlineKeycap(_ label: String) -> some View {
        let isWide = label == "Space"
        return Text(label)
            .font(.system(size: isWide ? 10 : 13, weight: .semibold, design: .rounded))
            .foregroundStyle(JunoTheme.primaryText(scheme))
            .padding(.horizontal, isWide ? 8 : 0)
            .frame(minWidth: isWide ? 0 : 26, minHeight: 24)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(JunoTheme.cardBackground(scheme))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(JunoTheme.border(scheme).opacity(0.55), lineWidth: 0.6)
            )
    }

    private func inlineKeycapLabels(for pref: JunoShortcutPreference) -> [String] {
        switch pref {
        case .fn: return ["fn"]
        case .rightCommand: return ["⌘"]
        case .rightOption: return ["⌥"]
        case .optionSpace: return ["⌥", "Space"]
        case .controlSpace: return ["⌃", "Space"]
        }
    }

    private func inlineKeycapStrip(for pref: JunoShortcutPreference) -> some View {
        let labels = inlineKeycapLabels(for: pref)
        return HStack(spacing: 4) {
            ForEach(Array(labels.enumerated()), id: \.offset) { idx, l in
                if idx > 0 {
                    Text("+")
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
                inlineKeycap(l)
            }
        }
    }

    private func shortcutRow(pref: JunoShortcutPreference) -> some View {
        let isSelected = selected == pref
        return Button {
            selected = pref
            JunoShortcutPreference.stored = pref
        } label: {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .strokeBorder(isSelected ? JunoDesignTokens.accent : JunoDesignTokens.border,
                                      lineWidth: isSelected ? 2 : 1)
                        .frame(width: 20, height: 20)
                    if isSelected {
                        Circle().fill(JunoDesignTokens.accent).frame(width: 10, height: 10)
                    }
                }
                inlineKeycapStrip(for: pref)
                    .frame(minWidth: 80, alignment: .leading)
                Text(pref.displayName)
                    .font(.system(.callout, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                if pref == JunoShortcutPreference.defaultShortcut {
                    Text("Recommended")
                        .font(.system(size: 10, weight: .semibold, design: .rounded))
                        .foregroundStyle(isSelected ? JunoDesignTokens.accent : JunoTheme.secondaryText(scheme))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            Capsule()
                                .fill(isSelected ? JunoDesignTokens.accent.opacity(0.10) : JunoTheme.elevatedCard(scheme))
                        )
                }
                if let conflict = conflictNote(for: pref) {
                    HStack(spacing: 4) {
                        Image(systemName: "info.circle.fill")
                            .font(.system(size: 10, weight: .semibold))
                        Text(conflict)
                            .font(.system(size: 10, weight: .medium, design: .rounded))
                    }
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(Capsule().fill(JunoTheme.elevatedCard(scheme).opacity(0.7)))
                }
                Spacer()
            }
            .padding(.horizontal, 14).padding(.vertical, 11)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(isSelected
                          ? JunoDesignTokens.accent.opacity(0.08)
                          : JunoTheme.elevatedCard(scheme))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(
                        isSelected ? JunoDesignTokens.accent.opacity(0.32) : JunoTheme.border(scheme).opacity(0.24),
                        lineWidth: isSelected ? 1 : 0.5
                    )
            )
        }
        .buttonStyle(.plain)
    }

    /// Soft, non-blocking hint about likely conflicts with macOS defaults.
    /// Never disables the option — the user can pick anything; we just
    /// flag the trade-off so they're not surprised.
    private func conflictNote(for pref: JunoShortcutPreference) -> String? {
        switch pref {
        case .fn: return nil
        case .rightCommand, .rightOption: return nil
        case .optionSpace: return "May overlap with Spotlight alternatives"
        case .controlSpace: return "Often used by input switchers"
        }
    }
}

// MARK: - Step 4 hero

/// Hero for the Setup step. A single hairline ring slowly orbits around a
/// small comma mark while models warm/download; once everything is ready the
/// orbit settles into a closed circle. Monochrome — the ring uses primary
/// text color, not accent. Respects reduce-motion (static dotted arc).
private struct OnboardingOrbitHero: View {
    let allReady: Bool
    let busy: Bool
    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var rotation: Double = 0

    var body: some View {
        ZStack {
            // Outer faint guide ring — always present, gives the hero its silhouette.
            Circle()
                .strokeBorder(JunoTheme.border(scheme).opacity(0.18), lineWidth: 0.5)
                .frame(width: 188, height: 188)

            // Orbiting arc (or static dotted ring when ready / reduce-motion).
            Group {
                if allReady {
                    Circle()
                        .strokeBorder(JunoTheme.primaryText(scheme).opacity(0.78), lineWidth: 0.9)
                } else if reduceMotion {
                    Circle()
                        .trim(from: 0, to: 0.72)
                        .stroke(
                            JunoTheme.primaryText(scheme).opacity(0.55),
                            style: StrokeStyle(lineWidth: 0.9, dash: [3, 5])
                        )
                        .rotationEffect(.degrees(-90))
                } else {
                    Circle()
                        .trim(from: 0, to: 0.30)
                        .stroke(
                            JunoTheme.primaryText(scheme).opacity(0.55),
                            style: StrokeStyle(lineWidth: 0.9, lineCap: .round)
                        )
                        .rotationEffect(.degrees(rotation - 90))
                }
            }
            .frame(width: 188, height: 188)
            .animation(.easeInOut(duration: 0.4), value: allReady)

            // Inner disc + comma — the protagonist, quiet but present.
            Circle()
                .fill(JunoDesignTokens.iconBg)
                .frame(width: 96, height: 96)
            JunoCommaMark(color: .white, scale: 0.96)
                .frame(width: 44, height: 60)
        }
        .frame(width: 188, height: 188)
        .shadow(color: Color.black.opacity(scheme == .dark ? 0.40 : 0.10), radius: 18, y: 8)
        .accessibilityHidden(true)
        .onAppear {
            if !reduceMotion && !allReady {
                withAnimation(.linear(duration: 2.6).repeatForever(autoreverses: false)) {
                    rotation = 360
                }
            }
        }
    }
}

// MARK: - Step 4: Models / setup

private struct OnboardingSetupStep: View {
    @ObservedObject var setup: JunoSetupModel
    @State private var didAutoStartInstall = false
    @Environment(\.colorScheme) private var scheme

    private var laneItems: [JunoSetupLaneViewModel] {
        JunoSetupPresentation.laneItems(from: setup)
    }

    private var requiredLaneItems: [JunoSetupLaneViewModel] {
        laneItems.filter(\.required)
    }

    private var hfCacheReady: Bool {
        !requiredLaneItems.isEmpty && requiredLaneItems.allSatisfy(\.ready)
    }

    /// True only when the HF cache is hot AND the engine has finished
    /// warming its components into memory. This is the gate the user
    /// actually cares about — "is the app ready to use right now?"
    /// — and the one the Continue button is enabled on.
    private var actuallyReady: Bool {
        hfCacheReady && setup.engineWarmingState != "warming"
    }

    private var busy: Bool {
        setup.installState == "downloading" || setup.engineWarmingState == "warming"
    }

    /// Visual state of the setup step. One of these is rendered at a
    /// time; never two stacked. Earlier versions stacked a "ready" card
    /// and a "still warming" card together, telling the user the app
    /// was both ready and not ready in the same frame.
    private enum SetupVisualState: Equatable {
        case brokerUnreachable
        case installFailed(String)
        case downloading
        case warmingEngine
        case ready
        case preparing  // catch-all for "we just got here, sorting things out"
    }

    private var visualState: SetupVisualState {
        if setup.installState.hasPrefix("failed") {
            return .installFailed(installFailureMessage)
        }
        if setup.installState == "broker_unreachable" {
            return .brokerUnreachable
        }
        if setup.installState == "downloading" {
            return .downloading
        }
        if hfCacheReady && setup.engineWarmingState == "warming" {
            return .warmingEngine
        }
        if hfCacheReady {
            return .ready
        }
        return .preparing
    }

    /// User-facing reason for an install failure, branched on the broker's
    /// install_state. Previously every failure showed "check your connection",
    /// which is misleading when the real cause is disk space.
    private var installFailureMessage: String {
        if setup.installState == "failed:insufficient_disk" {
            return "Your Mac is low on storage. Free up some space and try again."
        }
        return "Check your connection and try again."
    }

    /// Auto-start install when we have a reachable engine that hasn't downloaded yet.
    /// `onChange(of:)` does NOT fire for the initial value, so we also check on appear.
    private func autoStartIfNeeded() {
        guard !didAutoStartInstall, setup.canInstall else { return }
        didAutoStartInstall = true
        setup.triggerInstall()
    }

    var body: some View {
        let state = visualState
        return VStack(alignment: .center, spacing: 18) {
            OnboardingOrbitHero(allReady: state == .ready, busy: busy)
                .padding(.top, 4)

            VStack(spacing: 6) {
                Text(headerTitle(for: state))
                    .font(.system(size: 30, weight: .semibold, design: .rounded))
                    .tracking(-0.3)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(headerSubtitle(for: state))
                    .font(.system(size: 15, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: 460)

            // Single state-driven card. Replaces the earlier setup of
            // setupSummaryCard + engineWarmingCard + engineNotRunningCard
            // stacked together, which could surface contradictory states
            // (e.g. "Ready on this Mac" alongside "Setting up voice engine").
            setupStateCard(for: state)
                .frame(maxWidth: 540)

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .onAppear {
            setup.refresh()
            autoStartIfNeeded()
        }
        .onChange(of: setup.canInstall) { _ in autoStartIfNeeded() }
    }

    private func headerTitle(for state: SetupVisualState) -> String {
        switch state {
        case .ready:            return "Ready when you are"
        case .downloading:      return "Downloading voice models"
        case .warmingEngine:    return "Warming voice engine"
        case .brokerUnreachable:return "Engine isn't running"
        case .installFailed:    return setup.installState == "failed:insufficient_disk" ? "Not enough storage" : "Download didn't finish"
        case .preparing:        return "Preparing voice models"
        }
    }

    private func headerSubtitle(for state: SetupVisualState) -> String {
        switch state {
        case .ready:
            return "Models live on this Mac. Nothing leaves."
        case .downloading, .preparing:
            return "One-time download. Stays local once it's done."
        case .warmingEngine:
            return "Loading the models into memory — usually a few seconds."
        case .brokerUnreachable:
            return "Juno needs the local voice engine running to set up."
        case .installFailed(let message):
            return message
        }
    }

    // MARK: - State card (single source of truth for visible setup state)

    @ViewBuilder
    private func setupStateCard(for state: SetupVisualState) -> some View {
        OnboardingSurface {
            switch state {
            case .ready:
                readyCardBody
            case .downloading:
                downloadingCardBody
            case .warmingEngine:
                warmingCardBody
            case .brokerUnreachable:
                brokerUnreachableCardBody
            case .installFailed(let err):
                installFailedCardBody(error: err)
            case .preparing:
                preparingCardBody
            }
        }
    }

    /// "Ready" — checkmark, one-line confirmation, no progress bar.
    private var readyCardBody: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(JunoDesignTokens.meadow.opacity(0.13))
                    .frame(width: 40, height: 40)
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(JunoDesignTokens.meadow)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Ready on this Mac.")
                    .font(.system(.headline, design: .rounded).weight(.semibold))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("Everything Juno needs is downloaded and warmed up.")
                    .font(.system(.callout, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
    }

    /// "Downloading" — bytes/speed/ETA progress (the premium UI).
    private var downloadingCardBody: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(JunoDesignTokens.accent.opacity(0.12))
                        .frame(width: 40, height: 40)
                    Image(systemName: "arrow.down.circle.fill")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(JunoDesignTokens.accent)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Downloading voice models")
                        .font(.system(.headline, design: .rounded).weight(.semibold))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text("This runs once. Keep this window open.")
                        .font(.system(.callout, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            if setup.downloadActive && setup.downloadBytesTotal > 0 {
                downloadProgressDetails
            } else {
                downloadProgressIndeterminate
            }
            if let repo = setup.downloadCurrentRepo, setup.downloadReposTotal > 0 {
                currentModelLine(repo: repo)
            }
            if !setup.downloadLog.isEmpty {
                downloadStatusLog
            }
        }
    }

    /// "Now: <model> (2 of 4)" — which artifact the installer is on.
    private func currentModelLine(repo: String) -> some View {
        let position = min(setup.downloadReposDone + 1, max(setup.downloadReposTotal, 1))
        return HStack(spacing: 5) {
            Image(systemName: "shippingbox")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(JunoDesignTokens.accent)
            Text("Now: \(Self.shortModelName(repo)) (\(position) of \(setup.downloadReposTotal))")
                .font(.system(size: 11.5, weight: .medium, design: .rounded))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .lineLimit(1)
            Spacer(minLength: 0)
        }
    }

    /// Last few broker install-log lines — the user-visible answer to
    /// "is it actually doing anything?", and the breadcrumb we ask for
    /// when someone reports a stuck setup.
    private var downloadStatusLog: some View {
        let lines = Array(setup.downloadLog.suffix(3))
        return VStack(alignment: .leading, spacing: 3) {
            ForEach(Array(lines.enumerated()), id: \.offset) { index, line in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Circle()
                        .fill(JunoTheme.secondaryText(scheme).opacity(index == lines.count - 1 ? 0.8 : 0.35))
                        .frame(width: 4, height: 4)
                        .padding(.top, 1)
                    Text(line)
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundStyle(
                            JunoTheme.secondaryText(scheme)
                                .opacity(index == lines.count - 1 ? 1.0 : 0.55)
                        )
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
        }
        .padding(.top, 2)
    }

    private static func shortModelName(_ repo: String) -> String {
        repo.split(separator: "/").last.map(String.init) ?? repo
    }

    /// "Warming engine" — models on disk, MLX kernels loading into memory.
    /// Brief state on first launch after install; subsequent launches skip
    /// the visible warming state because the engine starts warm.
    private var warmingCardBody: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(JunoDesignTokens.accent.opacity(0.12))
                    .frame(width: 40, height: 40)
                ProgressView()
                    .controlSize(.small)
                    .tint(JunoDesignTokens.accent)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Loading models into memory")
                    .font(.system(.headline, design: .rounded).weight(.semibold))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("Almost there — a few more seconds.")
                    .font(.system(.callout, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
    }

    /// "Broker unreachable" — the engine subprocess isn't running.
    /// User can launch it manually; auto-launch lives elsewhere.
    private var brokerUnreachableCardBody: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(JunoDesignTokens.danger.opacity(0.10))
                        .frame(width: 40, height: 40)
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(JunoDesignTokens.danger)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Voice engine isn't running")
                        .font(.system(.headline, design: .rounded).weight(.semibold))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text("Juno needs the local engine running before models can be installed.")
                        .font(.system(.callout, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            Button("Launch engine") {
                setup.installAndStartLaunchdEngine()
                setup.refresh()
            }
            .controlSize(.regular)
            .buttonStyle(JunoPrimaryActionButtonStyle())
            .junoNoFocusRing()
            .frame(maxWidth: .infinity)
        }
    }

    /// "Install failed" — show the error briefly, offer a retry.
    private func installFailedCardBody(error: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(JunoDesignTokens.danger.opacity(0.10))
                        .frame(width: 40, height: 40)
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(JunoDesignTokens.danger)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Download didn't finish")
                        .font(.system(.headline, design: .rounded).weight(.semibold))
                        .foregroundStyle(JunoTheme.primaryText(scheme))
                    Text(error)
                        .font(.system(.callout, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            Button("Try again") { setup.triggerRepair() }
                .controlSize(.regular)
                .buttonStyle(JunoPrimaryActionButtonStyle())
                .junoNoFocusRing()
                .frame(maxWidth: .infinity)
        }
    }

    /// "Preparing" — broker is reachable, install hasn't started, or
    /// we're in the brief setup-status probe window. Indeterminate
    /// spinner; no error, no action needed.
    private var preparingCardBody: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(JunoDesignTokens.accent.opacity(0.12))
                    .frame(width: 40, height: 40)
                ProgressView()
                    .controlSize(.small)
                    .tint(JunoDesignTokens.accent)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Getting ready")
                    .font(.system(.headline, design: .rounded).weight(.semibold))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("Checking what's already on this Mac.")
                    .font(.system(.callout, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
    }

    // MARK: Download progress — premium bytes/speed/ETA UI.
    //
    // The broker publishes a rolling snapshot of bytes_so_far / bytes_total
    // / bytes_per_second / eta_seconds while a model-provisioning install
    // runs (see ``_poll_setup_install_progress`` server-side). This view
    // renders that data so onboarding shows real progress, not a
    // featureless spinner.

    @ViewBuilder
    private var downloadProgressDetails: some View {
        let total = setup.downloadBytesTotal
        let soFar = min(setup.downloadBytesSoFar, total)
        let fraction: Double = total > 0 ? Double(soFar) / Double(total) : 0
        VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: fraction)
                .tint(JunoDesignTokens.accent)
                .progressViewStyle(.linear)
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("\(Self.formatBytes(soFar)) / \(Self.formatBytes(total))")
                    .font(.system(size: 11.5, weight: .semibold, design: .monospaced))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("(\(Int((fraction * 100).rounded()))%)")
                    .font(.system(size: 11.5, weight: .regular, design: .monospaced))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                Text("\u{00B7}")
                    .font(.system(size: 11, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.55))
                Text(Self.formatSpeed(setup.downloadBytesPerSecond))
                    .font(.system(size: 11.5, weight: .regular, design: .monospaced))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                Spacer(minLength: 0)
                Text(Self.formatEta(setup.downloadEtaSeconds))
                    .font(.system(size: 11.5, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }
        }
    }

    @ViewBuilder
    private var downloadProgressIndeterminate: some View {
        VStack(alignment: .leading, spacing: 6) {
            ProgressView()
                .tint(JunoDesignTokens.accent)
                .progressViewStyle(.linear)
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(setup.downloadBytesSoFar > 0
                     ? "\(Self.formatBytes(setup.downloadBytesSoFar)) downloaded"
                     : "Preparing download…")
                    .font(.system(size: 11.5, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                Spacer(minLength: 0)
                if setup.downloadBytesPerSecond > 0 {
                    Text(Self.formatSpeed(setup.downloadBytesPerSecond))
                        .font(.system(size: 11.5, weight: .regular, design: .monospaced))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
            }
        }
    }

    private static func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        formatter.zeroPadsFractionDigits = false
        return formatter.string(fromByteCount: max(bytes, 0))
    }

    private static func formatSpeed(_ bytesPerSecond: Double) -> String {
        guard bytesPerSecond > 0 else { return "" }
        let perSec = Int64(bytesPerSecond.rounded())
        let s = formatBytes(perSec)
        return "\(s)/s"
    }

    private static func formatEta(_ seconds: Double?) -> String {
        guard let seconds, seconds > 0 else { return "" }
        if seconds < 5 {
            return "Almost done"
        }
        if seconds < 60 {
            return "About \(Int(seconds.rounded())) sec left"
        }
        let minutes = seconds / 60
        if minutes < 90 {
            let mInt = Int(minutes.rounded())
            return "About \(mInt) min left"
        }
        let hours = minutes / 60
        return String(format: "About %.1f hr left", hours)
    }

}

// MARK: - Step 5: Voice Actions intro
//
// "Juno can do things too" — introduces the killer feature: compound
// utterances that produce multiple outcomes in a single dictation. The
// step is required to *view* (you can't skip past it in the sequence)
// but the action-permission grant is opt-in via a "Maybe later" link.
//
// The hero visualization is the user's flagship moment: one spoken line
// → two result cards (a note + a reminder), connected by a soft "Y"
// fork so it reads as ONE input producing TWO outcomes. Below the hero
// is a smaller "you can also" cluster with single-action examples to
// show range without overshadowing the compound demo.

private struct OnboardingVoiceActionsStep: View {
    @Environment(\.colorScheme) private var scheme
    @State private var heroVisible = false
    @State private var resultsVisible = false
    @State private var moreVisible = false
    @StateObject private var actionPerms = JunoActionPermissionStore.shared

    // MARK: Rotating hero examples — only verbs/outcomes Juno actually does today.
    //
    // Each example uses a verb in ``juno_core_v3/actions/grammar.py`` so a
    // user repeating any of them will trigger the action. Outcomes describe
    // the literal effect:
        //   - Reminder → row in Reminders
    //   - Alarm     → 1-minute Calendar event with an alert (NOT a time block)
        //   - Note      → entry in a "Juno" folder in Notes
    //
    // Multi-action capability is real (the grammar parser returns a list of
    // actions from one utterance) and is surfaced honestly in the caption
    // below the hero — no faked compound demos.

    private struct ActionExample {
        let utterance: String
        let descriptor: JunoActionDescriptor
        let title: String
        let detail: String
    }

    private static let actionExamples: [ActionExample] = [
        ActionExample(
            utterance: "Hey Juno, remind me to call Sarah at 4pm tomorrow.",
            descriptor: .reminder,
            title: "Call Sarah",
            detail: "Reminders · Tomorrow, 4:00 PM"
        ),
        ActionExample(
            utterance: "Hey Juno, set an alarm for 7am tomorrow.",
            descriptor: .alarm,
            title: "Alarm",
            detail: "Alarm · Tomorrow, 7:00 AM"
        ),
        ActionExample(
            utterance: "Hey Juno, take a note: revisit the pricing tiers next week.",
            descriptor: .note,
            title: "Revisit pricing tiers",
            detail: "Saved to Notes"
        )
    ]

    @State private var exampleIndex: Int = 0
    @State private var rotationTask: DispatchWorkItem?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Drives the guided permission strip. Each transition replaces the
    /// CTA area in-place so the user always has context for the system
    /// popup that's about to (or just did) fire.
    private enum PermissionPhase: Equatable {
        case idle
        case askingReminders
        case remindersResolved(granted: Bool)
        case askingCalendar
        case calendarResolved(granted: Bool)
        case askingNotes
        case notesResolved(granted: Bool)
        case done(reminders: Bool, calendar: Bool, notes: Bool)
    }
    @State private var phase: PermissionPhase = .idle

    /// True once the user has tapped either "Enable Voice Actions" or
    /// "Maybe later". Parent reads this to know whether the user already
    /// made a decision (so the Continue/skip footer can re-render).
    var didDecide: Bool {
        JunoUserDefaults.actionsOnboardingDecisionMade
    }

    let onDecisionMade: () -> Void

    var body: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(spacing: JunoUI.Spacing.l) {
                header
                    .opacity(heroVisible ? 1 : 0)
                    .offset(y: heroVisible ? 0 : 8)
                    .animation(.easeOut(duration: 0.38), value: heroVisible)

                rotatingHero
                    .opacity(heroVisible ? 1 : 0)
                    .offset(y: heroVisible ? 0 : 10)
                    .animation(.easeOut(duration: 0.42).delay(0.12), value: heroVisible)

                accessPreview
                    .opacity(moreVisible ? 1 : 0)
                    .offset(y: moreVisible ? 0 : 8)
                    .animation(.easeOut(duration: 0.36).delay(0.20), value: moreVisible)

                actionsFooter
                    .opacity(moreVisible ? 1 : 0)
                    .animation(.easeOut(duration: 0.34).delay(0.28), value: moreVisible)

                Spacer(minLength: 0)
            }
            .padding(.vertical, 4)
            .frame(maxWidth: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .onAppear {
            heroVisible = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.20) {
                resultsVisible = true
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.42) {
                moreVisible = true
                if !reduceMotion {
                    scheduleNextExample()
                }
            }
            // Surfacing the action permission store on this step makes
            // sure the status reflects any grant the user gave before
            // entering onboarding (e.g. re-running onboarding).
            actionPerms.beginObserving()
        }
        .onDisappear {
            actionPerms.endObserving()
            rotationTask?.cancel()
            rotationTask = nil
        }
    }

    // MARK: Header

    private var header: some View {
        // Just the title and an honest one-liner. The hero card below shows
        // what the system does — no need to also describe it in prose up here.
        VStack(spacing: 4) {
            Text("Voice Actions")
                .font(.system(size: 26, weight: .semibold, design: .rounded))
                .tracking(-0.3)
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .multilineTextAlignment(.center)
            Text("Speak naturally — Juno turns the right phrases into reminders, alarms, and notes.")
                .font(.system(size: 13.5, weight: .regular, design: .rounded))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .multilineTextAlignment(.center)
                .lineSpacing(1)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: 520)
    }

    // MARK: Rotating hero — one natural utterance → one outcome, cycling.
    //
    // The previous version showed a single contrived compound sentence
    // ("...note the Sarah follow-up, remind me tomorrow morning, and
    // block 9 AM") with a trident fork to three result cards. It looked
    // dramatic but felt dishonest — people don't actually speak that
    // sentence. The current shape cycles three single-action utterances
    // through one transcript card and one outcome card. Range is implied
    // by rotation; each example is something a real user would actually say.

    private var rotatingHero: some View {
        let example = Self.actionExamples[exampleIndex]
        return VStack(spacing: 10) {
            spokenCard(for: example)
                .frame(maxWidth: 520)

            connectorChevron
                .frame(height: 14)

            outcomeCard(for: example)
                .frame(maxWidth: 520)
                .opacity(resultsVisible ? 1 : 0)
                .offset(y: resultsVisible ? 0 : 6)
                .animation(.easeOut(duration: 0.4), value: resultsVisible)

            // Three small dots so the user knows the demo will cycle.
            // No tap target on purpose — this is a non-interactive
            // progress affordance, not a control.
            rotationDots
                .padding(.top, 2)
        }
    }

    private func spokenCard(for example: ActionExample) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                miniMicGlyph
                Text("YOU SAY")
                    .junoType(.eyebrow)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                Spacer(minLength: 0)
            }
            Text("\u{201C}\(example.utterance)\u{201D}")
                .font(.system(size: 16, weight: .medium, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .fixedSize(horizontal: false, vertical: true)
                .lineSpacing(2)
                .multilineTextAlignment(.leading)
                .id(example.utterance) // crossfade the text when the example rotates
                .transition(.opacity)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .premiumCard()
    }

    private var miniMicGlyph: some View {
        ZStack {
            Circle()
                .fill(JunoDesignTokens.iconBg)
                .frame(width: 22, height: 22)
            Image(systemName: "mic.fill")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.white)
        }
    }

    /// A short downward-pointing chevron between the spoken card and the
    /// outcome card. Replaces the trident fork from the old hero — the
    /// chevron honestly reads as "one input → one output" rather than the
    /// fork's faked "one input → three outputs."
    private var connectorChevron: some View {
        Image(systemName: "chevron.down")
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.55))
            .accessibilityHidden(true)
    }

    private func outcomeCard(for example: ActionExample) -> some View {
        let d = example.descriptor
        return HStack(alignment: .center, spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(d.accent.opacity(0.14))
                    .frame(width: 44, height: 44)
                JunoActionNativeIcon(kind: d.kind, size: 36, fallbackColor: d.accent)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(d.displayName.uppercased())
                    .junoType(.eyebrow)
                    .foregroundStyle(d.accent.opacity(0.95))
                Text(example.title)
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
                Text(example.detail)
                    .font(.system(size: 12.5, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(JunoTheme.cardBackground(scheme))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme).opacity(0.42), lineWidth: 0.6)
        )
        .shadow(color: Color.black.opacity(scheme == .dark ? 0.22 : 0.05), radius: 6, y: 2)
        .id(d.displayName) // crossfade the whole card on rotate
        .transition(.opacity.combined(with: .move(edge: .bottom)))
    }

    private var rotationDots: some View {
        HStack(spacing: 7) {
            ForEach(Self.actionExamples.indices, id: \.self) { i in
                Circle()
                    .fill(JunoTheme.primaryText(scheme).opacity(i == exampleIndex ? 0.78 : 0.22))
                    .frame(width: 5, height: 5)
                    .animation(.easeOut(duration: 0.22), value: exampleIndex)
            }
        }
        .accessibilityHidden(true)
    }

    private func scheduleNextExample() {
        rotationTask?.cancel()
        let task = DispatchWorkItem { advanceExample() }
        rotationTask = task
        // 3.4s dwell — long enough to read the sentence and absorb the
        // outcome card, short enough that the rotation feels alive.
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.4, execute: task)
    }

    private func advanceExample() {
        withAnimation(.easeInOut(duration: 0.35)) {
            exampleIndex = (exampleIndex + 1) % Self.actionExamples.count
        }
        scheduleNextExample()
    }

    // MARK: Access preview — what each permission is for, in plain language.
    //
    // The previous inline-chip row ("Reminders | Alarm | Notes")
    // told the user WHICH permissions Juno asks for but not WHY each one.
    // A 3-row "why" list answers the question users actually have when a
    // permission dialog about to fire: "what is this for?" Each row also
    // gains a small status dot once granted, so the same surface doubles
    // as a recap on subsequent visits.

    private var accessPreview: some View {
        VStack(alignment: .leading, spacing: 0) {
            accessRow(
                actionKind: .reminder,
                label: "Reminders",
                why: "Saves a to-do when you say \u{201C}remind me\u{2026}\u{201D}",
                kind: .reminders
            )
            accessRowDivider
            accessRow(
                actionKind: .alarm,
                label: "Alarm",
                // Phrased honestly: Juno creates a 1-minute Calendar event
                // with an alert — never a multi-hour time block. The
                // permission is for Calendar because that's what macOS
                // gates, but the user-facing action is "alarm".
                why: "Sets an alarm at the time you say (a Calendar alert that rings even when Juno is closed).",
                kind: .calendarEvents
            )
            accessRowDivider
            accessRow(
                actionKind: .note,
                label: "Notes",
                why: "Captures freeform notes when you say \u{201C}take a note\u{2026}\u{201D}",
                kind: .notesAutomation
            )
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 4)
        .frame(maxWidth: 560)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme).opacity(0.55))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme).opacity(0.30), lineWidth: 0.5)
        )
    }

    private var accessRowDivider: some View {
        Rectangle()
            .fill(JunoTheme.border(scheme).opacity(0.20))
            .frame(height: 0.5)
            .padding(.leading, 38)
    }

    private func accessRow(
        actionKind: JunoActionKind,
        label: String,
        why: String,
        kind: JunoActionPermissionDescriptor
    ) -> some View {
        let granted = actionPerms.status(for: kind) == .granted
        let descriptor = actionKind.descriptor
        return HStack(alignment: .center, spacing: 12) {
            JunoActionNativeIconTile(kind: actionKind, tileSize: 26, iconSize: 22, fallbackTint: descriptor.accent)
            VStack(alignment: .leading, spacing: 1) {
                Text(label)
                    .font(.system(size: 12.5, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(why)
                    .font(.system(size: 11.5, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            if granted {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(JunoDesignTokens.meadow)
                    .accessibilityLabel(Text("\(label) allowed"))
            }
        }
        .padding(.vertical, 9)
    }

    // MARK: Footer — primary CTA, guided permission strip, skip + caption

    private var actionsFooter: some View {
        // Onboarding is mandatory: every step must complete. There is no
        // "Maybe later" escape hatch here — the user runs the permission
        // flow, decides per-prompt (allow or deny is fine; both count as
        // a decision), then the footer Continue button unlocks.
        VStack(spacing: 8) {
            if didDecide {
                decisionMadeBadge
            } else if phase == .idle {
                Button {
                    startEnableFlow()
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "bolt.fill")
                            .font(.system(size: 12, weight: .semibold))
                        Text("Grant all three")
                            .font(.system(size: 14, weight: .semibold, design: .rounded))
                    }
                    .padding(.horizontal, 18)
                    .padding(.vertical, 4)
                }
                .controlSize(.regular)
                .buttonStyle(JunoPrimaryActionButtonStyle())
                .junoNoFocusRing()
            } else {
                guidedPermissionStrip
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .frame(maxWidth: 460)
        .animation(.easeInOut(duration: 0.22), value: phase)
    }

    /// In-place replacement for the CTA while permissions are being asked.
    /// Each phase narrates the system popup that just fired (or is firing
    /// next), so the user always has context. Chrome stays in the same
    /// vertical slot as the button so the page doesn't reflow. When a
    /// permission is denied, a small "Open Settings" affordance is shown
    /// inline so the user can fix it without having to leave the flow.
    private var guidedPermissionStrip: some View {
        let (icon, tint, message) = stripContents(for: phase)
        return HStack(spacing: 10) {
            stripIcon(symbol: icon, tint: tint)
            Text(message)
                .font(.system(size: 13, weight: .medium, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .fixedSize(horizontal: false, vertical: true)
                .multilineTextAlignment(.leading)
            Spacer(minLength: 0)
            if let retry = retryActionForCurrentPhase() {
                Button(retry.label) { retry.action() }
                    .buttonStyle(.plain)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoDesignTokens.accent)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(
                        Capsule().fill(JunoDesignTokens.accent.opacity(0.10))
                    )
                    .overlay(
                        Capsule().strokeBorder(JunoDesignTokens.accent.opacity(0.32), lineWidth: 0.5)
                    )
                    .junoNoFocusRing()
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme).opacity(0.9))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(tint.opacity(0.32), lineWidth: 0.6)
        )
    }

    /// Returns a retry affordance for the current phase when something
    /// went wrong (denied permission). Otherwise nil — most phases are
    /// either spinning or successful and don't need a button.
    private func retryActionForCurrentPhase() -> (label: String, action: () -> Void)? {
        switch phase {
        case .remindersResolved(granted: false):
            return ("Open Reminders", { actionPerms.openRemindersSettings() })
        case .calendarResolved(granted: false):
            return ("Open Calendar", { actionPerms.openCalendarSettings() })
        case .notesResolved(granted: false):
            return ("Open Notes access", { JunoSystemSettingsLinks.openAutomationPrivacy() })
        default:
            return nil
        }
    }

    @ViewBuilder
    private func stripIcon(symbol: String, tint: Color) -> some View {
        ZStack {
            Circle()
                .fill(tint.opacity(0.16))
                .frame(width: 22, height: 22)
            if symbol == "progress" {
                ProgressView()
                    .controlSize(.mini)
                    .scaleEffect(0.8)
            } else {
                Image(systemName: symbol)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(tint)
            }
        }
    }

    /// Returns (icon, tint, message) for the current phase. Icon == "progress"
    /// is the sentinel for "render a ProgressView spinner instead of an
    /// SF Symbol" — keeps the strip layout identical across spinning and
    /// resolved states. Copy is intentionally plain — earlier versions used
    /// tech-speak like "Asking macOS for X permission"; users read that as
    /// a log line, not a status.
    private func stripContents(for phase: PermissionPhase) -> (String, Color, String) {
        switch phase {
        case .idle:
            return ("bolt.fill", JunoDesignTokens.accent, "")
        case .askingReminders:
            return ("progress", JunoDesignTokens.accent,
                    "Look for the Reminders prompt.")
        case .remindersResolved(let granted):
            return granted
                ? ("checkmark", JunoDesignTokens.meadow, "Reminders allowed.")
                : ("xmark", JunoDesignTokens.danger,
                   "Reminders isn't allowed. Change this anytime.")
        case .askingCalendar:
            return ("progress", JunoDesignTokens.accent,
                    "Look for the Calendar prompt.")
        case .calendarResolved(let granted):
            return granted
                ? ("checkmark", JunoDesignTokens.meadow, "Calendar allowed.")
                : ("xmark", JunoDesignTokens.danger,
                   "Calendar isn't allowed. Change this anytime.")
        case .askingNotes:
            return ("progress", JunoDesignTokens.accent,
                    "Look for the Notes automation prompt.")
        case .notesResolved(let granted):
            return granted
                ? ("checkmark", JunoDesignTokens.meadow, "Notes allowed.")
                : ("xmark", JunoDesignTokens.danger,
                   "Notes isn't allowed. Change this anytime.")
        case .done(let r, let c, let n):
            if r || c || n {
                return ("checkmark", JunoDesignTokens.meadow,
                        "Voice Actions are on.")
            } else {
                return ("moon.zzz.fill", JunoTheme.secondaryText(scheme),
                        "Voice Actions stay off for now.")
            }
        }
    }

    private var decisionMadeBadge: some View {
        // After the user has decided, replace the CTA with a quiet
        // confirmation row so the page doesn't sit on a "click me" CTA
        // that does nothing. If actions are OFF, offer a way back — the
        // user shouldn't have to discover Settings to flip their mind.
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: JunoUserDefaults.actionsEnabled ? "checkmark.circle.fill" : "moon.zzz.fill")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(JunoUserDefaults.actionsEnabled ? JunoDesignTokens.meadow : JunoTheme.secondaryText(scheme))
                Text(JunoUserDefaults.actionsEnabled
                     ? "Voice Actions are on."
                     : "Voice Actions are off for now.")
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(
                Capsule().fill(JunoTheme.elevatedCard(scheme).opacity(0.85))
            )
            .overlay(
                Capsule().strokeBorder(JunoTheme.border(scheme).opacity(0.35), lineWidth: 0.5)
            )

            if !JunoUserDefaults.actionsEnabled {
                Button("Change my mind") {
                    // Reset the decision and the phase so the user can run
                    // the guided flow again without leaving onboarding.
                    JunoUserDefaults.actionsOnboardingDecisionMade = false
                    phase = .idle
                }
                .buttonStyle(.plain)
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .foregroundStyle(JunoDesignTokens.accent)
                .junoNoFocusRing()
            }
        }
    }

    // MARK: Decision handlers — phase-driven guided permission flow.
    //
    // The flow narrates each macOS popup the user is about to see, then
    // confirms the outcome before moving to the next. We never block on a
    // permission outcome — the step always advances. Notes Automation
    // consent is surfaced by sending one harmless AppleEvent at Notes —
    // see JunoActionPermissionStore.requestNotesAutomation.

    /// Entry point from the "Enable Voice Actions" button. Decides which
    /// phase to enter first based on what's already been determined, then
    /// hands off to phase-specific request helpers.
    ///
    /// IMPORTANT: this does NOT flip ``actionsEnabled`` or
    /// ``actionsOnboardingDecisionMade``. Flipping them here would render
    /// the "✓ Voice Actions are on" badge *during* the first popup —
    /// before the user has answered anything — which is misleading. Both
    /// flags are set inside ``scheduleAfterNotes`` once all three prompts
    /// have resolved.
    private func startEnableFlow() {
        let remindersDecided = actionPerms.status(for: .reminders) != .notDetermined
        if remindersDecided {
            // Treat already-determined as a resolved phase so the strip
            // still narrates the state for the user instead of skipping
            // silently.
            let granted = actionPerms.status(for: .reminders) == .granted
            phase = .remindersResolved(granted: granted)
            scheduleAfterReminders()
        } else {
            phase = .askingReminders
            actionPerms.requestReminders { status in
                DispatchQueue.main.async {
                    phase = .remindersResolved(granted: status == .granted)
                    scheduleAfterReminders()
                }
            }
        }
    }

    /// After the reminders phase resolves, dwell briefly so the user can
    /// read the outcome, then move on to the calendar request. A denied
    /// outcome dwells a bit longer so the inline "Open Reminders" retry
    /// button is reachable before the next prompt fires.
    private func scheduleAfterReminders() {
        let dwell: TimeInterval = isDeniedPhase(phase) ? 1.6 : 0.4
        DispatchQueue.main.asyncAfter(deadline: .now() + dwell) {
            requestCalendarPhase()
        }
    }

    private func requestCalendarPhase() {
        let calendarDecided = actionPerms.status(for: .calendarEvents) != .notDetermined
        if calendarDecided {
            let granted = actionPerms.status(for: .calendarEvents) == .granted
            phase = .calendarResolved(granted: granted)
            scheduleAfterCalendar()
        } else {
            phase = .askingCalendar
            actionPerms.requestCalendarEvents { status in
                DispatchQueue.main.async {
                    phase = .calendarResolved(granted: status == .granted)
                    scheduleAfterCalendar()
                }
            }
        }
    }

    private func scheduleAfterCalendar() {
        let dwell: TimeInterval = isDeniedPhase(phase) ? 1.6 : 0.4
        DispatchQueue.main.asyncAfter(deadline: .now() + dwell) {
            requestNotesPhase()
        }
    }

    private func requestNotesPhase() {
        let notesDecided = actionPerms.status(for: .notesAutomation) != .notDetermined
        if notesDecided {
            let granted = actionPerms.status(for: .notesAutomation) == .granted
            phase = .notesResolved(granted: granted)
            scheduleAfterNotes()
        } else {
            phase = .askingNotes
            actionPerms.requestNotesAutomation { status in
                DispatchQueue.main.async {
                    phase = .notesResolved(granted: status == .granted)
                    scheduleAfterNotes()
                }
            }
        }
    }

    private func scheduleAfterNotes() {
        let dwell: TimeInterval = isDeniedPhase(phase) ? 1.6 : 0.4
        DispatchQueue.main.asyncAfter(deadline: .now() + dwell) {
            let rGranted = actionPerms.status(for: .reminders) == .granted
            let cGranted = actionPerms.status(for: .calendarEvents) == .granted
            let nGranted = actionPerms.status(for: .notesAutomation) == .granted
            // Now — and only now — flip the persisted flags. The user has
            // answered all three prompts, so the badge that reads these
            // values will represent reality. ``actionsEnabled`` follows
            // the same rule as the Actions page: at least one grant turns
            // the feature on.
            JunoUserDefaults.actionsEnabled = rGranted || cGranted || nGranted
            JunoUserDefaults.actionsOnboardingDecisionMade = true
            phase = .done(reminders: rGranted, calendar: cGranted, notes: nGranted)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.7) {
                onDecisionMade()
            }
        }
    }

    /// True when the phase represents a denied resolution — used to give
    /// the inline retry button enough on-screen time to be clickable.
    private func isDeniedPhase(_ p: PermissionPhase) -> Bool {
        switch p {
        case .remindersResolved(granted: false),
             .calendarResolved(granted: false),
             .notesResolved(granted: false):
            return true
        default:
            return false
        }
    }

}

// MARK: - Step 6: Ready

private struct OnboardingReadyStep: View {
    @State private var shown = false
    @State private var phase: HUDDemoPhase = .idle
    @State private var typedCount: Int = 0
    @State private var loopTask: DispatchWorkItem?
    @State private var sentenceIndex: Int = 0
    @State private var underlineProgress: CGFloat = 0
    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private enum HUDDemoPhase { case idle, listening, settled }

    /// Three real-feeling examples that imply Juno's range without spelling
    /// it out: a quick work nudge, a scheduling sentence with mixed
    /// punctuation, and a capture-style note. The user sees one full cycle
    /// before the loop repeats — long enough to read "this fits my day."
    private let demoSentences: [String] = [
        "Send the updated timeline by Friday.",
        "Push tomorrow's standup to 4pm — I'll be on a flight.",
        "Take a note: revisit the pricing tiers next week."
    ]

    private var demoSentence: String { demoSentences[sentenceIndex] }

    /// Closes the loop with step 0: if the user typed a name there, we
    /// greet them by it. Falls back to a clean impersonal headline if not.
    private var personalisedHeadline: String {
        let name = JunoUserDefaults.preferredDisplayName?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if name.isEmpty { return "You're set up." }
        return "You're set up, \(name)."
    }

    var body: some View {
        VStack(spacing: 22) {
            Spacer(minLength: 0)

            VStack(spacing: 6) {
                Text(personalisedHeadline)
                    .font(.system(size: 32, weight: .semibold, design: .rounded))
                    .tracking(-0.3)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("This is what your first line will feel like.")
                    .font(.system(size: 15, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }
            .opacity(shown ? 1 : 0)
            .offset(y: shown ? 0 : 8)
            .animation(.easeOut(duration: 0.38), value: shown)

            hudStage
                .opacity(shown ? 1 : 0)
                .offset(y: shown ? 0 : 12)
                .animation(.easeOut(duration: 0.42).delay(0.08), value: shown)

            quickRef
                .opacity(shown ? 1 : 0)
                .offset(y: shown ? 0 : 12)
                .animation(.easeOut(duration: 0.42).delay(0.18), value: shown)

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            shown = true
            if reduceMotion {
                phase = .settled
                typedCount = demoSentence.count
                underlineProgress = 1
            } else {
                runLoop()
            }
        }
        .onDisappear {
            loopTask?.cancel()
            loopTask = nil
        }
    }

    // MARK: - HUD stage (animated)

    private var hudStage: some View {
        VStack(spacing: 14) {
            // Mini mock of the dictation HUD pill — same dark capsule, same
            // Juno mark, but compressed for onboarding so it reads as
            // "this is what you'll see" not as a real HUD.
            HStack(spacing: 14) {
                miniMark
                    .frame(width: 30, height: 30)
                waveform
                    .frame(height: 22)
                    .frame(maxWidth: .infinity, alignment: .leading)
                phaseBadge
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
            .frame(maxWidth: 540)
            .background(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(JunoDesignTokens.iconBg)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.08), lineWidth: 0.8)
            )
            .shadow(color: Color.black.opacity(0.34), radius: 18, y: 8)

            // Typing transcript. Reveals one character at a time during
            // ``listening``, holds during ``settled``, blanks during
            // ``idle``. The accent underline on the trailing caret sells
            // the live-transcription feel.
            transcriptCard
                .frame(maxWidth: 540)
        }
    }

    private var miniMark: some View {
        ZStack {
            Circle()
                .fill(Color.white.opacity(0.06))
            JunoCommaMark(color: .white, scale: 0.96)
                .frame(width: 14, height: 18)
                .scaleEffect(phase == .listening ? 1.06 : 1.0)
                .animation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true), value: phase)
        }
    }

    private var waveform: some View {
        // Six bars that pulse during listening and flatten otherwise. No
        // dependency on the real audio engine — purely a visual cue that
        // says "Juno is hearing you."
        HStack(spacing: 4) {
            ForEach(0..<6, id: \.self) { i in
                Capsule()
                    .fill(Color.white.opacity(phase == .listening ? 0.85 : 0.18))
                    .frame(width: 3, height: barHeight(for: i))
                    .animation(
                        .easeInOut(duration: 0.42)
                        .repeatForever(autoreverses: true)
                        .delay(Double(i) * 0.06),
                        value: phase
                    )
            }
        }
    }

    private func barHeight(for i: Int) -> CGFloat {
        switch phase {
        case .idle: return 4
        case .settled: return 6
        case .listening:
            let pattern: [CGFloat] = [10, 18, 12, 22, 14, 8]
            return pattern[i % pattern.count]
        }
    }

    private var phaseBadge: some View {
        // Monochrome state. The dot is always white-on-ink — its presence
        // (open ring vs filled vs hairline) carries the state, not its hue.
        // Label drops to lowercase so it reads like a quiet caption rather
        // than a system log. ALL-CAPS monospaced felt very 2014-developer.
        HStack(spacing: 6) {
            phaseDot
            Text(phaseLabel)
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(Color.white.opacity(0.72))
        }
    }

    private var phaseDot: some View {
        Group {
            switch phase {
            case .idle:
                Circle()
                    .strokeBorder(Color.white.opacity(0.45), lineWidth: 0.8)
            case .listening:
                Circle()
                    .fill(Color.white.opacity(0.92))
            case .settled:
                Circle()
                    .strokeBorder(Color.white.opacity(0.92), lineWidth: 0.8)
            }
        }
        .frame(width: 6, height: 6)
        .animation(.easeOut(duration: 0.22), value: phase)
    }

    private var phaseLabel: String {
        // "Pasted" is the literal action the user sees when a dictation
        // lands — Juno places the polished sentence into the active field.
        // "Landed" was poetic but vague; "Pasted" maps to what's happening.
        switch phase {
        case .idle: return "Ready"
        case .listening: return "Listening"
        case .settled: return "Pasted"
        }
    }

    private var transcriptCard: some View {
        let revealed = String(demoSentence.prefix(typedCount))
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 10) {
                Text(revealed.isEmpty ? " " : revealed)
                    .font(.system(size: 17, weight: .medium, design: .rounded))
                    .foregroundStyle(phase == .settled ? JunoTheme.primaryText(scheme) : JunoTheme.secondaryText(scheme))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .lineLimit(1)
                    .truncationMode(.tail)
                if phase == .listening {
                    // Monochrome caret — primary text colour, not accent.
                    Rectangle()
                        .fill(JunoTheme.primaryText(scheme).opacity(0.78))
                        .frame(width: 2, height: 18)
                        .transition(.opacity)
                }
            }
            // Hairline that draws itself left-to-right when the line "lands."
            // Replaces the green-circle checkmark with a typographic gesture
            // — the same metaphor as ink settling on paper.
            GeometryReader { geo in
                Rectangle()
                    .fill(JunoTheme.primaryText(scheme).opacity(0.78))
                    .frame(width: geo.size.width * underlineProgress, height: 1)
            }
            .frame(height: 1)
            .opacity(phase == .settled ? 1 : 0)
            .animation(.easeOut(duration: 0.22), value: phase)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(JunoTheme.cardBackground(scheme))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme).opacity(0.5), lineWidth: 0.7)
        )
    }

    // MARK: - Quick reference

    private var quickRef: some View {
        // Spec-sheet calm: thin numerals, hairline rules, no card chrome.
        // Drops the heavy accent bubble numbers and divider lines, which
        // read as a kid-school tutorial. The sheet now sits on the page
        // background, separated from the demo above only by space.
        VStack(spacing: 0) {
            quickRefStep(number: 1, glyph: nil, text: "Place your cursor anywhere you write.")
            quickRefDivider
            quickRefStep(number: 2, glyph: shortcutGlyph, text: "Press once to start dictating.")
            quickRefDivider
            quickRefStep(number: 3, glyph: nil, text: "Speak naturally. Press again — Juno lands the polished sentence.")
        }
        .frame(maxWidth: 540)
    }

    private var shortcutGlyph: String {
        switch JunoShortcutPreference.stored {
        case .fn: return "fn"
        case .rightCommand: return "⌘"
        case .rightOption: return "⌥"
        case .optionSpace: return "⌥+␣"
        case .controlSpace: return "⌃+␣"
        }
    }

    private var quickRefDivider: some View {
        Rectangle()
            .fill(JunoTheme.border(scheme).opacity(0.30))
            .frame(height: 0.5)
    }

    private func quickRefStep(number: Int, glyph: String?, text: String) -> some View {
        HStack(alignment: .center, spacing: 16) {
            // Thin ink numeral — replaces the blue-bubble badge with the
            // kind of figure you'd see in a system spec table.
            Text("\(number)")
                .font(.system(size: 16, weight: .light, design: .rounded))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .frame(width: 18, alignment: .leading)
            Text(text)
                .font(.system(size: 13, weight: .medium, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 8)
            if let g = glyph {
                Text(g)
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(JunoTheme.elevatedCard(scheme))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .strokeBorder(JunoTheme.border(scheme).opacity(0.45), lineWidth: 0.5)
                    )
            }
        }
        .padding(.vertical, 12)
    }

    // MARK: - Demo loop

    /// Cycles idle → listening (with character-by-character reveal) →
    /// settled → idle. Self-rescheduling via DispatchWorkItem so we
    /// cleanly cancel on disappear and respect the reduce-motion guard.
    private func runLoop() {
        loopTask?.cancel()
        let task = DispatchWorkItem { tickStart() }
        loopTask = task
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4, execute: task)
    }

    private func tickStart() {
        withAnimation(.easeOut(duration: 0.2)) {
            phase = .listening
            typedCount = 0
            underlineProgress = 0
        }
        let total = demoSentence.count
        let stepInterval: TimeInterval = 0.045
        for i in 1...total {
            let when = DispatchTime.now() + .milliseconds(Int(Double(i) * stepInterval * 1000))
            DispatchQueue.main.asyncAfter(deadline: when) {
                guard self.phase == .listening else { return }
                self.typedCount = i
            }
        }
        let listenSeconds = Double(total) * stepInterval + 0.4
        DispatchQueue.main.asyncAfter(deadline: .now() + listenSeconds) {
            withAnimation(.spring(response: 0.32, dampingFraction: 0.78)) {
                phase = .settled
            }
            // Draw the underline a beat after the words settle — the line
            // chasing the ink is what sells the "this just landed" feeling.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.18) {
                withAnimation(.easeOut(duration: 0.42)) {
                    underlineProgress = 1
                }
            }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + listenSeconds + 1.6) {
            withAnimation(.easeIn(duration: 0.25)) {
                phase = .idle
                typedCount = 0
                underlineProgress = 0
            }
            // Cycle to the next sentence so the loop never repeats verbatim
            // — three different examples imply "Juno fits any line you'd
            // dictate" without saying it out loud.
            sentenceIndex = (sentenceIndex + 1) % demoSentences.count
            let next = DispatchWorkItem { tickStart() }
            loopTask = next
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6, execute: next)
        }
    }
}

// MARK: - Flow container

private enum JunoOnboardingChrome {
    static let horizontalPadding: CGFloat = 28
    static let titlebarClearance: CGFloat = 42
    static let progressHeight: CGFloat = 34
    static let progressBottomGap: CGFloat = 16
    static let footerHeight: CGFloat = 64
    static let bottomPadding: CGFloat = 10
}

private struct JunoOnboardingView: View {
    let onFinish: () -> Void
    @StateObject private var setup = JunoSetupModel()
    @ObservedObject private var perms = JunoPermissionMonitor.shared
    @State private var step = 0
    @State private var preferredNameDraft: String = JunoUserDefaults.preferredDisplayName ?? ""
    // @AppStorage so step 4's gate flips reactively the moment the actions
    // flow finishes — UserDefaults reads from a computed property won't
    // trigger SwiftUI updates on their own.
    @AppStorage(JunoUserDefaults.actionsOnboardingDecisionMadeKey) private var actionsDecisionMade: Bool = false
    @FocusState private var primaryActionFocused: Bool
    @Environment(\.colorScheme) private var scheme

    private let totalSteps = 6

    private var setupRequiredModelsReady: Bool {
        let required = JunoSetupPresentation.laneItems(from: setup).filter(\.required)
        return !required.isEmpty && required.allSatisfy(\.ready)
    }

    var body: some View {
        VStack(spacing: 0) {
            progressBar
                .frame(height: JunoOnboardingChrome.progressHeight, alignment: .top)
                .padding(.bottom, JunoOnboardingChrome.progressBottomGap)

            // No outer ScrollView. Each step is responsible for fitting in
            // the available chrome — a fixed-height onboarding feels more
            // crafted than scrolling cards, and avoids the "step 1 has more
            // stuff than fits" trap from the previous version.
            Group {
                switch step {
                case 0:
                    OnboardingIntroStep(
                        preferredName: $preferredNameDraft,
                        onSubmit: { advanceFromCurrentStep() }
                    )
                case 1:
                    OnboardingPermissionsStep(onRefresh: { perms.refresh() })
                case 2:
                    OnboardingActivationStep()
                case 3:
                    OnboardingSetupStep(setup: setup)
                case 4:
                    OnboardingVoiceActionsStep(onDecisionMade: {
                        // Refresh focus on the footer Continue once the user
                        // resolves the in-step CTA so Return advances cleanly.
                        focusPrimaryIfAppropriate()
                    })
                case 5:
                    OnboardingReadyStep()
                default:
                    EmptyView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .id(step)
            .transition(.asymmetric(
                insertion: .move(edge: .trailing).combined(with: .opacity),
                removal:   .move(edge: .leading).combined(with: .opacity)
            ))
            .animation(.spring(response: 0.38, dampingFraction: 0.88), value: step)

            footerBar
        }
        .padding(.horizontal, JunoOnboardingChrome.horizontalPadding)
        .padding(.top, JunoOnboardingChrome.titlebarClearance)
        .padding(.bottom, JunoOnboardingChrome.bottomPadding)
        .frame(minWidth: 720, minHeight: 660)
        .groupBoxStyle(JunoBrandGroupBoxStyle())
        .junoBrandWindow()
        .onAppear {
            perms.refresh()
            JunoPermissionMonitor.shared.startMonitoring()
            setup.startPolling()
            focusPrimaryIfAppropriate()
        }
        .onDisappear { setup.stopPolling() }
        .onChange(of: step) { _ in focusPrimaryIfAppropriate() }
        .onChange(of: nextDisabled) { disabled in
            // Step 3: Continue flips from "Please wait…" to "Continue" when
            // the download completes — pull focus so Return advances.
            if !disabled { focusPrimaryIfAppropriate() }
        }
        .onChange(of: perms.micStatus) { _ in focusPrimaryIfAppropriate() }
        .onChange(of: perms.axGranted) { _ in focusPrimaryIfAppropriate() }
    }

    // MARK: Progress bar

    private var footerBar: some View {
        HStack(spacing: 12) {
            if step > 0 {
                Button("Back") { withAnimation { step -= 1 } }
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
            }
            Spacer()
            Text(stepDescriptor)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .tracking(1.0)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
            if step < totalSteps - 1 {
                if nextDisabled, let waitingLabel = waitingIndicatorLabel {
                    // Step 3 (Setup) while downloading, warming, or
                    // starting. Render a calm progress indicator in the
                    // footer instead of a darkened "Please wait…" button
                    // — a button-shaped primary CTA invites the click
                    // that does nothing.
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                            .tint(JunoDesignTokens.accent)
                        Text(waitingLabel)
                            .font(.system(size: 12.5, weight: .medium, design: .rounded))
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                    }
                } else {
                    // Step 4 (Actions): Continue is disabled until the
                    // guided permission flow has completed. Without an
                    // inline hint the user has no idea why the button is
                    // greyed out, especially keyboard-only users who
                    // pressed Return and got nothing.
                    if step == 4 && nextDisabled {
                        Text("Use Grant all three above first — Continue unlocks after.")
                            .font(.system(size: 11.5, design: .rounded))
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .padding(.trailing, 4)
                    }
                    Button(nextLabel) { advanceFromCurrentStep() }
                    .keyboardShortcut(.defaultAction)
                    .controlSize(.regular)
                    .buttonStyle(JunoPrimaryActionButtonStyle())
                    .junoNoFocusRing()
                    .focused($primaryActionFocused)
                    .disabled(nextDisabled)
                }
            } else {
                Button("Open Juno") {
                    JunoMilestoneNotifier.shared.playOnboardingHeroIfNeeded()
                    JunoOnboardingDefaults.isCompleted = true
                    // Voice Actions is now an explicit opt-in on the
                    // "Juno can do things too" step (#4). If the user
                    // got that far without making a decision (e.g. they
                    // pressed Back to skip), treat it as deferred so
                    // the post-onboarding nudge can fire after a few
                    // dictations.
                    if !JunoUserDefaults.actionsOnboardingDecisionMade {
                        JunoUserDefaults.actionsOnboardingDecisionMade = true
                    }
                    JunoShellRuntime.shared.startHotkeyBridge()
                    JunoEngineLifecycle.shared.boot()
                    syncOnboardingPersonalizationToBroker()
                    NotificationCenter.default.post(name: .junoOpenMainWindow, object: nil)
                    JunoShellWindowOpener.showMainWindow(section: .home)
                    DispatchQueue.main.async {
                        onFinish()
                    }
                }
                .keyboardShortcut(.defaultAction)
                .controlSize(.regular)
                .buttonStyle(JunoPrimaryActionButtonStyle())
                .junoNoFocusRing()
                .focused($primaryActionFocused)
            }
        }
        .padding(.horizontal, 24)
        .frame(height: JunoOnboardingChrome.footerHeight)
        .background(JunoTheme.windowBackground(scheme).opacity(0.96))
        .overlay(alignment: .top) {
            Rectangle()
                .fill(JunoUI.hairline(.faint, scheme: scheme))
                .frame(height: 1)
        }
    }

    private var progressBar: some View {
        // Hairline rail + filled segment, eyebrow labels — calmer than the
        // previous accent-shadow dots. The active step's label uses the
        // primary ink colour; future steps fade. No per-segment glow.
        HStack(spacing: JunoUI.Spacing.s) {
            ForEach(0..<totalSteps, id: \.self) { i in
                VStack(alignment: .leading, spacing: JunoUI.Spacing.xs) {
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule()
                                .fill(JunoUI.hairline(.regular, scheme: scheme))
                                .frame(height: 2)
                            Capsule()
                                .fill(i <= step ? JunoDesignTokens.accent : Color.clear)
                                .frame(width: i <= step ? geo.size.width : 0, height: 2)
                                .animation(JunoUI.Motion.cardReveal, value: step)
                        }
                    }
                    .frame(height: 2)
                    JunoEyebrow(
                        text: stepTitle(for: i),
                        color: i <= step
                            ? JunoTheme.primaryText(scheme).opacity(0.78)
                            : JunoTheme.secondaryText(scheme).opacity(0.42)
                    )
                }
            }
        }
        .padding(.horizontal, 2)
    }

    private var stepDescriptor: String {
        "STEP \(step + 1) OF \(totalSteps)"
    }

    private func stepTitle(for index: Int) -> String {
        switch index {
        case 0: return "Welcome"
        case 1: return "Access"
        case 2: return "Shortcut"
        case 3: return "Setup"
        case 4: return "Actions"
        case 5: return "Try it"
        default: return ""
        }
    }

    private var nextLabel: String {
        if step == 1 {
            switch permissionsCascade {
            case .requestMic: return "Allow microphone"
            case .openMicSettings: return "Open Microphone privacy"
            case .openAccessibility: return "Open Accessibility"
            case .advance: return "Continue"
            }
        }
        // Every other step uses the same neutral label. The previous
        // "Finish setup" string on the Setup step was misleading — Setup
        // is one of six screens, not the end of the flow.
        return "Continue"
    }

    /// When the footer's Continue is in a waiting-but-disabled state, the
    /// footer renders a non-clickable progress indicator instead of a
    /// darkened button. Returns the label to show next to the spinner.
    /// ``nil`` means "render the regular Button" (enabled or merely
    /// visually disabled).
    ///
    /// Step 3 (Setup) is never allowed to fall through to a darkened
    /// Continue: every non-ready substate maps to a friendly progress
    /// indicator. This kept users from clicking a button that does
    /// nothing while models prepared.
    private var waitingIndicatorLabel: String? {
        if step == 3 {
            if setup.installState == "downloading" {
                return "Downloading models…"
            }
            if setup.engineWarmingState == "warming" {
                return "Warming up…"
            }
            if !setupRequiredModelsReady {
                // brokerUnreachable, installFailed, or preparing — any
                // transient state where the engine isn't ready and we're
                // not yet showing a more specific progress message.
                return "Starting up…"
            }
        }
        return nil
    }

    private var nextDisabled: Bool {
        // Step 1 (Permissions) is never disabled — the Continue button cascades
        // through the next pending permission action so Return alone can walk
        // the user from "nothing granted" to "all granted → advance" without
        // ever leaving the keyboard.
        if step == 3 {
            // Block until the engine is *actually* ready — HF cache hot AND
            // models loaded into memory. Without the warming check, a fresh
            // install would let the user click Continue mid-warmup and land
            // on the next step with an unwarmed engine.
            if !setupRequiredModelsReady { return true }
            if setup.engineWarmingState == "warming" { return true }
            if setup.installState == "downloading" { return true }
            return false
        }
        // Step 4 (Actions) is now mandatory: the user must run the guided
        // permission flow (allowing or denying each prompt — both count) so
        // we never ship a half-set-up app. ``actionsDecisionMade`` is an
        // @AppStorage so the flip from false→true triggers a re-eval.
        if step == 4 { return !actionsDecisionMade }
        return false
    }

    private enum PermissionsCascadeAction {
        case requestMic
        case openMicSettings
        case openAccessibility
        case advance
    }

    /// Picks the next pending permission action for step 1. Mic comes first
    /// (notDetermined → request, denied → open Settings), then Accessibility,
    /// then advance. Visible screen text stays optional and never blocks.
    private var permissionsCascade: PermissionsCascadeAction {
        switch perms.micStatus {
        case .notDetermined: return .requestMic
        case .denied, .restricted: return .openMicSettings
        case .authorized:
            return perms.axGranted ? .advance : .openAccessibility
        @unknown default: return .openMicSettings
        }
    }

    /// Single source of truth for advancing past the current step. Used by the
    /// Continue button and by Return-on-Submit from inside step views (e.g. the
    /// name TextField on the intro step), so keyboard-only users can walk the
    /// whole onboarding by pressing Return repeatedly.
    private func advanceFromCurrentStep() {
        guard step < totalSteps - 1 else { return }
        guard !nextDisabled else { return }
        if step == 0 {
            let t = preferredNameDraft.trimmingCharacters(in: .whitespacesAndNewlines)
            JunoUserDefaults.preferredDisplayName = t.isEmpty ? nil : t
            startRuntimeAfterIntro()
            withAnimation { step += 1 }
            return
        }
        if step == 1 {
            switch permissionsCascade {
            case .requestMic:
                perms.requestMic { _ in perms.refresh() }
            case .openMicSettings:
                perms.openMicSettings()
            case .openAccessibility:
                perms.openAccessibilitySettings()
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { perms.refresh() }
            case .advance:
                perms.refresh()
                withAnimation { step += 1 }
            }
            return
        }
        withAnimation { step += 1 }
    }

    /// Move keyboard focus to the primary footer button when it's the right
    /// answer for this step. Skipped on step 0 because the name TextField owns
    /// focus there — Return on the field already calls advance via onSubmit.
    private func focusPrimaryIfAppropriate() {
        guard step != 0 else { return }
        guard !nextDisabled else { return }
        DispatchQueue.main.async { primaryActionFocused = true }
    }

    private func startRuntimeAfterIntro() {
        Task { @MainActor in
            JunoEngineLifecycle.shared.boot()
        }
    }

    private func syncOnboardingPersonalizationToBroker() {
        let name = (JunoUserDefaults.preferredDisplayName ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return }

        // Fire-and-forget: onboarding should never block on broker availability.
        JunoBroker.post(
            path: "api/broker/personalization/user_profile",
            payload: ["display_name": name]
        )
        JunoBroker.post(
            path: "api/broker/memory/vocab",
            payload: [
                "term": name,
                "canonical_form": name,
                "boost": 2.0,
            ]
        )
    }
}

// MARK: - Window host

private final class JunoOnboardingCloseDelegate: NSObject, NSWindowDelegate {
    private let onTeardown: () -> Void
    init(onTeardown: @escaping () -> Void) { self.onTeardown = onTeardown }
    func windowWillClose(_ notification: Notification) {
        if let window = notification.object as? NSWindow {
            window.contentViewController = nil
            window.contentView = NSView()
        }
        // The success path ("Open Juno and try it") already posts
        // ``.junoOpenMainWindow`` itself, *after* setting
        // ``onboardingCompleted = true``. If we also post here on
        // window-close, the X (red traffic-light) button enters a
        // re-entrant loop:
        //   X → windowWillClose → .junoOpenMainWindow →
        //   ``JunoMainWindow.show`` (line ~3425) sees onboarding
        //   incomplete → calls ``showIfNeeded`` → onboarding pops back.
        // Result: the user perceives the close button as broken.
        // Only post when onboarding genuinely completed; otherwise let
        // the window quietly dismiss (the app stays in the menu bar,
        // and Dock-reopen will surface onboarding again via
        // ``applicationShouldHandleReopen``).
        if JunoOnboardingDefaults.isCompleted {
            NotificationCenter.default.post(name: .junoOpenMainWindow, object: nil)
        }
        onTeardown()
    }
}

enum JunoOnboardingWindow {
    private static var windowController: NSWindowController?
    private static var closeDelegate: JunoOnboardingCloseDelegate?

    @MainActor
    static func showIfNeeded() {
        guard !JunoOnboardingDefaults.isCompleted else { return }
        if let wc = windowController, let w = wc.window, w.isVisible {
            w.appearance = NSAppearance(named: .aqua)
            w.contentView?.appearance = NSAppearance(named: .aqua)
            JunoWindowActivation.bringToFront(w)
            return
        }
        windowController = nil; closeDelegate = nil
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 760, height: 760),
            styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
            backing: .buffered, defer: false
        )
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.appearance = NSAppearance(named: .aqua)
        // Onboarding is the fixed first-run product experience. Keep it in
        // the default Light appearance even if an existing user has switched
        // the rest of Juno to Dark.
        let root = JunoOnboardingView { window.close() }
            .preferredColorScheme(.light)
        window.contentViewController = NSHostingController(rootView: root)
        window.contentView?.appearance = NSAppearance(named: .aqua)
        window.title = "Welcome to Juno"
        // Lock the content size BEFORE centering. NSHostingController will
        // otherwise resize the window to fit its SwiftUI ideal size after
        // ``center()`` runs, leaving the window pinned to whatever origin
        // ``center()`` picked for the old (smaller) frame — which lands it
        // off-center, often hugging the right edge of the screen.
        window.setContentSize(NSSize(width: 760, height: 760))
        window.minSize = NSSize(width: 720, height: 700)
        window.center()
        window.isReleasedWhenClosed = false
        // Standard window level so System Settings (a normal-level
        // window) sits above Juno when the user goes to grant
        // Accessibility / Microphone — otherwise the onboarding window
        // covers the toggle and the user can't reach it.
        window.level = .normal
        window.collectionBehavior = [.moveToActiveSpace, .fullScreenAuxiliary]
        let del = JunoOnboardingCloseDelegate {
            JunoOnboardingWindow.windowController = nil
            JunoOnboardingWindow.closeDelegate = nil
        }
        closeDelegate = del; window.delegate = del
        let wc = NSWindowController(window: window)
        windowController = wc
        wc.showWindow(nil)
        JunoWindowActivation.bringToFront(window)
    }

    /// Re-opens the welcome flow from the menu and **clears** the completed flag so the user can run through again.
    @MainActor
    static func show() {
        JunoOnboardingDefaults.isCompleted = false
        windowController?.close()
        windowController = nil; closeDelegate = nil
        showIfNeeded()
    }
}

// MARK: - Data flow diagram (Step 4)
//
// A small three-chip illustration: speech goes into the mic, gets
// transcribed by the on-device model, and lands at your cursor. The
// middle chip is labeled "On this Mac" so the privacy promise reads
// as *spatial*, not just textual. Three SF Symbols + two hairline
// arrows; no Lottie / image asset shipped.
private struct OnboardingDataFlowDiagram: View {
    @Environment(\.colorScheme) private var scheme
    @State private var animationPhase: Int = 0

    var body: some View {
        HStack(alignment: .center, spacing: JunoUI.Spacing.xs) {
            chip(symbol: "mic.fill", caption: "You speak", lit: animationPhase >= 0)
            arrow(lit: animationPhase >= 1)
            chip(
                symbol: "cpu",
                caption: "On this Mac",
                lit: animationPhase >= 1,
                emphasised: true
            )
            arrow(lit: animationPhase >= 2)
            chip(symbol: "text.cursor", caption: "Your cursor", lit: animationPhase >= 2)
        }
        .frame(maxWidth: .infinity)
        .onAppear {
            // Three-step ambient sweep, then settle. Repeats every 4s as a
            // quiet idle motion so the surface feels alive without being noisy.
            startSweep()
        }
    }

    private func startSweep() {
        animationPhase = 0
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.30) {
            withAnimation(JunoUI.Motion.cardReveal) { animationPhase = 1 }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.80) {
            withAnimation(JunoUI.Motion.cardReveal) { animationPhase = 2 }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 4.0) { startSweep() }
    }

    @ViewBuilder
    private func chip(
        symbol: String,
        caption: String,
        lit: Bool,
        emphasised: Bool = false
    ) -> some View {
        VStack(spacing: JunoUI.Spacing.xs) {
            ZStack {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(chipFill(lit: lit, emphasised: emphasised))
                Image(systemName: symbol)
                    .font(.system(size: emphasised ? 18 : 16, weight: .semibold))
                    .foregroundStyle(chipForeground(lit: lit, emphasised: emphasised))
            }
            .frame(width: emphasised ? 56 : 48, height: emphasised ? 56 : 48)
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(JunoUI.hairline(.regular, scheme: scheme), lineWidth: 0.8)
            )
            Text(caption)
                .junoType(.caption)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .lineLimit(1)
        }
        .opacity(lit ? 1.0 : 0.55)
    }

    @ViewBuilder
    private func arrow(lit: Bool) -> some View {
        ZStack {
            Capsule()
                .fill(JunoUI.hairline(.regular, scheme: scheme))
                .frame(height: 1.2)
            Capsule()
                .fill(JunoDesignTokens.accent.opacity(lit ? 0.85 : 0))
                .frame(width: lit ? 22 : 0, height: 1.2)
                .animation(JunoUI.Motion.cardReveal, value: lit)
            Image(systemName: "chevron.right")
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(lit ? JunoDesignTokens.accent : JunoTheme.secondaryText(scheme).opacity(0.42))
                .offset(x: 12)
                .animation(JunoUI.Motion.cardReveal, value: lit)
        }
        .frame(width: 38, height: 14)
        .padding(.bottom, 18)   // align with chip body, ignore caption row
    }

    private func chipFill(lit: Bool, emphasised: Bool) -> Color {
        if emphasised && lit {
            return JunoDesignTokens.accent.opacity(scheme == .dark ? 0.18 : 0.10)
        }
        return JunoTheme.elevatedCard(scheme)
    }

    private func chipForeground(lit: Bool, emphasised: Bool) -> Color {
        if emphasised && lit { return JunoDesignTokens.accent }
        if !lit { return JunoTheme.secondaryText(scheme).opacity(0.55) }
        return JunoTheme.primaryText(scheme).opacity(0.85)
    }
}
