import AppKit
import SwiftUI

// MARK: - Shared chrome
//
// Custom Popover-based dropdown shell used throughout the app in place of
// the stock `Picker` / NSPopUpButton. Each picker on the Settings page (and
// the App / Writing-style pickers on Per-app writing) renders the same
// trigger anatomy:
//
//   ┌──────────────────────────┐
//   │ [leading]  Title    ⌃⌄   │   ← `JunoPopoverPickerTrigger`
//   └──────────────────────────┘
//
// and opens a popover whose rows share this anatomy:
//
//   ┌──────────────────────────────┐
//   │ [leading]  Title          ✓  │   ← `JunoPopoverPickerRow`
//   │            Subtitle          │
//   └──────────────────────────────┘
//
// Specializations differ only in their leading view (swatch, glyph, keycap
// strip, screen diagram). The popover panel itself is rendered by SwiftUI's
// `.popover()`; we provide an `arrowEdge: .bottom` anchor on macOS so the
// popover sits visually attached to the trigger.

/// Tap-to-open, popover-anchored Button. Replaces the `NSPopUpButton`
/// chevron-only stock dropdown.
private struct JunoPopoverPickerTrigger<Leading: View>: View {
    let title: String
    @ViewBuilder var leading: () -> Leading
    let isDisabled: Bool

    @Environment(\.colorScheme) private var scheme

    init(
        title: String,
        isDisabled: Bool = false,
        @ViewBuilder leading: @escaping () -> Leading
    ) {
        self.title = title
        self.isDisabled = isDisabled
        self.leading = leading
    }

    var body: some View {
        HStack(spacing: 8) {
            leading()
            Text(title)
                .font(.system(size: 13, weight: .regular, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme).opacity(isDisabled ? 0.45 : 1))
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer(minLength: 6)
            Image(systemName: "chevron.up.chevron.down")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(isDisabled ? 0.4 : 0.75))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .frame(minHeight: 28)
        .background(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme).opacity(isDisabled ? 0.5 : 1))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme).opacity(0.45), lineWidth: 0.5)
        )
        .contentShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
    }
}

/// One row inside a popover panel. Leading view + title + optional
/// subtitle + selection check + optional trailing accessory (e.g. a
/// "Recommended" pill or a conflict hint).
private struct JunoPopoverPickerRow<Leading: View, Accessory: View>: View {
    let title: String
    let subtitle: String?
    let isSelected: Bool
    @ViewBuilder var leading: () -> Leading
    @ViewBuilder var accessory: () -> Accessory

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            leading()
                .frame(width: 28, alignment: .center)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.system(size: 11, weight: .regular, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 8)
            accessory()
            Image(systemName: "checkmark")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(isSelected ? JunoDesignTokens.accent : Color.clear)
                .frame(width: 14)
        }
        .contentShape(Rectangle())
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
    }
}

extension JunoPopoverPickerRow where Accessory == EmptyView {
    init(
        title: String,
        subtitle: String?,
        isSelected: Bool,
        @ViewBuilder leading: @escaping () -> Leading
    ) {
        self.title = title
        self.subtitle = subtitle
        self.isSelected = isSelected
        self.leading = leading
        self.accessory = { EmptyView() }
    }
}

/// Press / hover background. Used on every Button inside a Juno popover.
struct JunoPopoverRowButtonStyle: ButtonStyle {
    @State private var hovering = false
    @Environment(\.colorScheme) private var scheme

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(
                        configuration.isPressed
                            ? JunoDesignTokens.accent.opacity(0.14)
                            : (hovering
                               ? JunoTheme.elevatedCard(scheme).opacity(0.85)
                               : Color.clear)
                    )
            )
            .contentShape(Rectangle())
            .onHover { hovering = $0 }
    }
}

extension View {
    /// Suppress the loud blue keyboard focus rectangle that macOS draws
    /// around the first row of a popover. Available on macOS 14+; older
    /// systems keep the default focus behaviour.
    @ViewBuilder
    func junoNoFocusEffect() -> some View {
        if #available(macOS 14.0, *) {
            self.focusEffectDisabled()
        } else {
            self
        }
    }
}

// MARK: - Inline keycaps (shared with onboarding Step 3)
//
// These render single-key and combo shortcut visuals consistently across
// the onboarding shortcut step and the Settings dropdown. Lifted into a
// shared helper so the two surfaces can never visually diverge.

/// A single keycap (e.g. `⌥`, `fn`, `Space`).
struct JunoInlineKeycap: View {
    let label: String
    var size: Size = .small
    @Environment(\.colorScheme) private var scheme

    enum Size {
        case small   // popover rows + dropdown trigger
        case medium  // onboarding inline rows

        var minWidth: CGFloat   { self == .small ? 24 : 26 }
        var minHeight: CGFloat  { self == .small ? 22 : 24 }
        var fontSize: CGFloat   { self == .small ? 12 : 13 }
        var widePad: CGFloat    { self == .small ? 7  : 8  }
        var wideFontSize: CGFloat { self == .small ? 9.5 : 10 }
    }

    var body: some View {
        let isWide = label == "Space"
        return Text(label)
            .font(.system(
                size: isWide ? size.wideFontSize : size.fontSize,
                weight: .semibold, design: .rounded
            ))
            .foregroundStyle(JunoTheme.primaryText(scheme))
            .padding(.horizontal, isWide ? size.widePad : 0)
            .frame(minWidth: isWide ? 0 : size.minWidth, minHeight: size.minHeight)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(JunoTheme.cardBackground(scheme))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(JunoTheme.border(scheme).opacity(0.55), lineWidth: 0.6)
            )
    }
}

/// A combo strip rendering one or more keycaps with `+` separators.
struct JunoInlineKeycapStrip: View {
    let labels: [String]
    var size: JunoInlineKeycap.Size = .small
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(spacing: 4) {
            ForEach(Array(labels.enumerated()), id: \.offset) { idx, l in
                if idx > 0 {
                    Text("+")
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
                JunoInlineKeycap(label: l, size: size)
            }
        }
    }
}

/// Canonical keycap labels for a shortcut preference. Single source of
/// truth used by both onboarding and Settings.
func junoInlineKeycapLabels(for pref: JunoShortcutPreference) -> [String] {
    switch pref {
    case .fn:           return ["fn"]
    case .rightCommand: return ["⌘"]
    case .rightOption:  return ["⌥"]
    case .optionSpace:  return ["⌥", "Space"]
    case .controlSpace: return ["⌃", "Space"]
    }
}

// MARK: - Leading visuals

/// Tiny half-tone swatch preview for the Appearance picker. Light is paper,
/// Dark is ink, System is split half-and-half so the user can see at a
/// glance which mode they're picking.
private struct AppearanceSwatch: View {
    let preference: JunoAppearancePreference
    var size: CGFloat = 22

    var body: some View {
        ZStack {
            switch preference {
            case .light:
                Circle()
                    .fill(JunoUI.Calm.paper)
                    .overlay(Circle().strokeBorder(JunoUI.Calm.inkSoft.opacity(0.25), lineWidth: 0.5))
            case .dark:
                Circle()
                    .fill(JunoUI.Calm.ink)
                    .overlay(Circle().strokeBorder(Color.white.opacity(0.18), lineWidth: 0.5))
            case .system:
                ZStack {
                    Circle()
                        .fill(JunoUI.Calm.paper)
                    Rectangle()
                        .fill(JunoUI.Calm.ink)
                        .frame(width: size / 2, height: size)
                        .offset(x: size / 4)
                        .clipShape(Circle())
                }
                .overlay(Circle().strokeBorder(Color.black.opacity(0.20), lineWidth: 0.5))
            }
        }
        .frame(width: size, height: size)
    }
}

/// Small "screen" rectangle with an accent pill rendered where the HUD
/// actually sits — top center or bottom — so the user sees the position
/// rather than reading the word.
private struct HUDPositionDiagram: View {
    let position: JunoUserDefaults.HUDPosition
    var width: CGFloat = 44
    var height: CGFloat = 28

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ZStack(alignment: position == .topCenter ? .top : .bottom) {
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme))
                .overlay(
                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                        .strokeBorder(JunoTheme.border(scheme).opacity(0.55), lineWidth: 0.5)
                )
            Capsule()
                .fill(JunoDesignTokens.accent)
                .frame(width: width * 0.4, height: 3)
                .padding(.top, position == .topCenter ? 4 : 0)
                .padding(.bottom, position == .bottomCenter ? 4 : 0)
        }
        .frame(width: width, height: height)
    }
}

/// SF-symbol-style glyph framed inside a faint card so it sits visually
/// alongside swatches / keycaps / diagrams without going visually quieter.
private struct GlyphLeading: View {
    let systemName: String
    var dim: Bool = false
    var accent: Bool = false

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme))
                .overlay(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .strokeBorder(JunoTheme.border(scheme).opacity(0.45), lineWidth: 0.5)
                )
            Image(systemName: systemName)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(
                    accent ? JunoDesignTokens.accent
                           : JunoTheme.secondaryText(scheme).opacity(dim ? 0.5 : 1)
                )
        }
        .frame(width: 24, height: 24)
    }
}

// MARK: - Specialized pickers

/// Appearance · Light / Dark / Match System with half-tone swatch previews.
struct AppearancePopoverPicker: View {
    @Binding var selection: JunoAppearancePreference
    var onPick: (JunoAppearancePreference) -> Void = { _ in }

    @State private var open = false
    @Environment(\.colorScheme) private var scheme

    private let rows: [(value: JunoAppearancePreference, subtitle: String)] = [
        (.light,  "Always light"),
        (.dark,   "Always dark"),
        (.system, "Match your Mac"),
    ]

    var body: some View {
        Button { open.toggle() } label: {
            JunoPopoverPickerTrigger(title: selection.title) {
                AppearanceSwatch(preference: selection, size: 16)
            }
        }
        .buttonStyle(.plain)
        .frame(maxWidth: 220)
        .popover(isPresented: $open, arrowEdge: .bottom) {
            VStack(spacing: 0) {
                ForEach(rows, id: \.value) { row in
                    Button {
                        selection = row.value
                        onPick(row.value)
                        open = false
                    } label: {
                        JunoPopoverPickerRow(
                            title: row.value.title,
                            subtitle: row.subtitle,
                            isSelected: selection == row.value
                        ) {
                            AppearanceSwatch(preference: row.value, size: 20)
                        }
                    }
                    .buttonStyle(JunoPopoverRowButtonStyle())
                    .junoNoFocusEffect()
                }
            }
            .padding(6)
            .frame(width: 240)
        }
    }
}

/// Shortcut · keycap-rendered rows. Mirrors the visual language of the
/// onboarding Step 3 screen.
struct ShortcutPopoverPicker: View {
    @Binding var selection: JunoShortcutPreference
    var onPick: (JunoShortcutPreference) -> Void = { _ in }

    @State private var open = false
    @Environment(\.colorScheme) private var scheme

    private func conflictNote(for pref: JunoShortcutPreference) -> String? {
        switch pref {
        case .optionSpace:  return "May overlap with Spotlight alternatives"
        case .controlSpace: return "Often used by input switchers"
        case .rightCommand, .rightOption: return nil
        case .fn:           return nil
        }
    }

    var body: some View {
        Button { open.toggle() } label: {
            JunoPopoverPickerTrigger(title: selection.displayName) {
                JunoInlineKeycapStrip(labels: junoInlineKeycapLabels(for: selection))
            }
        }
        .buttonStyle(.plain)
        .frame(maxWidth: 260)
        .popover(isPresented: $open, arrowEdge: .bottom) {
            VStack(spacing: 0) {
                ForEach(JunoShortcutPreference.allCases, id: \.self) { pref in
                    Button {
                        selection = pref
                        onPick(pref)
                        open = false
                    } label: {
                        JunoPopoverPickerRow(
                            title: pref.displayName,
                            subtitle: nil,
                            isSelected: selection == pref,
                            leading: {
                                JunoInlineKeycapStrip(labels: junoInlineKeycapLabels(for: pref))
                                    .frame(minWidth: 80, alignment: .leading)
                            },
                            accessory: {
                                Group {
                                    if pref == JunoShortcutPreference.defaultShortcut {
                                        Text("Recommended")
                                            .font(.system(size: 10, weight: .semibold, design: .rounded))
                                            .foregroundStyle(JunoDesignTokens.accent)
                                            .padding(.horizontal, 7)
                                            .padding(.vertical, 2)
                                            .background(
                                                Capsule().fill(JunoDesignTokens.accent.opacity(scheme == .dark ? 0.18 : 0.10))
                                            )
                                    } else if let note = conflictNote(for: pref) {
                                        HStack(spacing: 4) {
                                            Image(systemName: "info.circle")
                                                .font(.system(size: 10, weight: .semibold))
                                            Text(note)
                                                .font(.system(size: 10, weight: .medium, design: .rounded))
                                        }
                                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                                    }
                                }
                            }
                        )
                    }
                    .buttonStyle(JunoPopoverRowButtonStyle())
                    .junoNoFocusEffect()
                }
            }
            .padding(6)
            .frame(width: 380)
        }
    }
}

/// HUD position · tiny screen-with-pill diagram per option.
struct HUDPositionPopoverPicker: View {
    @Binding var selection: JunoUserDefaults.HUDPosition
    var onPick: (JunoUserDefaults.HUDPosition) -> Void = { _ in }

    @State private var open = false

    private let rows: [(value: JunoUserDefaults.HUDPosition, title: String, subtitle: String)] = [
        (.topCenter,    "Top center",     "Above your menu bar — the default"),
        (.bottomCenter, "Bottom",         "Out of the way of the menu bar or notch"),
    ]

    private func triggerTitle() -> String {
        rows.first(where: { $0.value == selection })?.title ?? selection.title
    }

    var body: some View {
        Button { open.toggle() } label: {
            JunoPopoverPickerTrigger(title: triggerTitle()) {
                HUDPositionDiagram(position: selection, width: 24, height: 16)
            }
        }
        .buttonStyle(.plain)
        .frame(maxWidth: 220)
        .popover(isPresented: $open, arrowEdge: .bottom) {
            VStack(spacing: 0) {
                ForEach(rows, id: \.value) { row in
                    Button {
                        selection = row.value
                        onPick(row.value)
                        open = false
                    } label: {
                        JunoPopoverPickerRow(
                            title: row.title,
                            subtitle: row.subtitle,
                            isSelected: selection == row.value
                        ) {
                            HUDPositionDiagram(position: row.value, width: 44, height: 28)
                        }
                    }
                    .buttonStyle(JunoPopoverRowButtonStyle())
                    .junoNoFocusEffect()
                }
            }
            .padding(6)
            .frame(width: 280)
        }
    }
}

/// Retention picker · used by both "Keep recordings for" (audio) and
/// "Keep history for" (history). One enum kind covers both — the cases
/// list and copy come from the existing ``JunoRetentionChoice`` cases.
struct RetentionPopoverPicker: View {
    enum Kind { case audio, history }

    @Binding var selection: JunoRetentionChoice
    let kind: Kind
    var isDisabled: Bool = false
    var onPick: (JunoRetentionChoice) -> Void = { _ in }

    @State private var open = false

    private var cases: [JunoRetentionChoice] {
        switch kind {
        case .audio:   return JunoRetentionChoice.audioCases
        case .history: return JunoRetentionChoice.historyCases
        }
    }

    private func subtitle(for choice: JunoRetentionChoice) -> String {
        switch (kind, choice.policy, choice.days) {
        case (.audio, "forever", _):   return "Keep every recording — never delete on its own"
        case (.audio, "off", _):       return "Drop audio as soon as the run finishes"
        case (.audio, "days", let d?): return "Recent \(d == 7 ? "week" : "\(d) days") only"
        case (.history, "forever", _): return "Keep every session — never delete on its own"
        case (.history, "off", _):     return "Don’t keep transcripts after the run finishes"
        case (.history, "days", let d?):
            if d == 30 { return "Recent month only" }
            if d == 90 { return "Recent three months only" }
            return "Recent \(d) days only"
        default: return ""
        }
    }

    private func glyph(for choice: JunoRetentionChoice) -> some View {
        switch (choice.policy, choice.days) {
        case ("forever", _): return AnyView(GlyphLeading(systemName: "infinity", accent: true))
        case ("off", _):     return AnyView(GlyphLeading(systemName: "nosign", dim: true))
        case ("days", _):    return AnyView(GlyphLeading(systemName: "clock"))
        default:             return AnyView(GlyphLeading(systemName: "clock"))
        }
    }

    var body: some View {
        Button {
            guard !isDisabled else { return }
            open.toggle()
        } label: {
            JunoPopoverPickerTrigger(title: selection.title, isDisabled: isDisabled) {
                glyph(for: selection)
            }
        }
        .buttonStyle(.plain)
        .frame(maxWidth: 220)
        .disabled(isDisabled)
        .popover(isPresented: $open, arrowEdge: .bottom) {
            VStack(spacing: 0) {
                ForEach(cases) { choice in
                    Button {
                        selection = choice
                        onPick(choice)
                        open = false
                    } label: {
                        JunoPopoverPickerRow(
                            title: choice.title,
                            subtitle: subtitle(for: choice),
                            isSelected: selection == choice
                        ) {
                            glyph(for: choice)
                        }
                    }
                    .buttonStyle(JunoPopoverRowButtonStyle())
                    .junoNoFocusEffect()
                }
            }
            .padding(6)
            .frame(width: 280)
        }
    }
}

/// Generic glyph + raw-String selection picker, used for Language and
/// Speaking environment. Caller supplies the option list and the leading
/// SF-symbol name per option.
struct GenericGlyphPopoverPicker: View {
    struct Option: Identifiable, Hashable {
        let value: String
        let title: String
        let subtitle: String
        let systemName: String
        var accent: Bool = false
        var id: String { value }
    }

    @Binding var selection: String
    let options: [Option]
    var triggerLeadingSystemName: String? = nil
    var onPick: (String) -> Void = { _ in }

    @State private var open = false

    private var selectedOption: Option? {
        options.first(where: { $0.value == selection })
    }

    private var triggerTitle: String {
        selectedOption?.title ?? "Select…"
    }

    var body: some View {
        Button { open.toggle() } label: {
            JunoPopoverPickerTrigger(title: triggerTitle) {
                GlyphLeading(
                    systemName: triggerLeadingSystemName
                        ?? selectedOption?.systemName
                        ?? "circle.dotted",
                    accent: selectedOption?.accent ?? false
                )
            }
        }
        .buttonStyle(.plain)
        .frame(maxWidth: 240)
        .popover(isPresented: $open, arrowEdge: .bottom) {
            VStack(spacing: 0) {
                ForEach(options) { option in
                    Button {
                        selection = option.value
                        onPick(option.value)
                        open = false
                    } label: {
                        JunoPopoverPickerRow(
                            title: option.title,
                            subtitle: option.subtitle,
                            isSelected: selection == option.value
                        ) {
                            GlyphLeading(systemName: option.systemName, accent: option.accent)
                        }
                    }
                    .buttonStyle(JunoPopoverRowButtonStyle())
                    .junoNoFocusEffect()
                }
            }
            .padding(6)
            .frame(width: 300)
        }
    }
}

// MARK: - Canonical option lists for Language + Environment

enum JunoLanguagePickerOptions {
    static let all: [GenericGlyphPopoverPicker.Option] = [
        .init(value: "auto",          title: "Auto-detect",      subtitle: "Juno detects the spoken language per session",       systemName: "globe",                       accent: true),
        .init(value: "en",            title: "English",          subtitle: "English only",                                       systemName: "character.bubble"),
        .init(value: "pair:en,hi",    title: "Hindi + English",  subtitle: "Hindi mixed with English",                            systemName: "character.bubble.fill"),
        .init(value: "zh",            title: "Mandarin Chinese", subtitle: "Mandarin Chinese only",                              systemName: "character.bubble"),
        .init(value: "es",            title: "Spanish",          subtitle: "Spanish only",                                       systemName: "character.bubble"),
        .init(value: "keep_original", title: "Keep original",    subtitle: "Don’t translate — keep what was spoken",             systemName: "quote.bubble"),
    ]
}
