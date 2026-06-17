import AppKit
import SwiftUI

// MARK: - App-wide theme

/// Maps brand tokens to surface roles for both light (paper) and dark (ink) modes.
///
/// **Main-window layout archetypes** (share spacing + header rhythm; keep per-page behavior):
/// - **Split list + detail** — ``NavigationSplitView`` first column: ``View/junoSplitPanePadding()``,
///   ``JunoSplitColumnTitleRow``, optional search row, list with ``View/junoSplitListChrome()``;
///   detail: scroll + ``View/junoDetailPagePadding()``. Examples: History, Modes, Memory list+detail.
/// - **Scroll form** — full-width ``ScrollView`` with grouped cards (Settings, Per-app writing).
/// - **Reading width** — ``View/junoCenteredReadingPane(maxWidth:)`` (Home, Settings).
enum JunoTheme {
    enum Density {
        static let controlHeight: CGFloat = 30
        static let compactRowMinHeight: CGFloat = 34
        static let listRowMinHeight: CGFloat = 38
        static let cardPadding: CGFloat = 14
        static let cardGap: CGFloat = 12
        static let itemGap: CGFloat = 8
        static let readingWidth: CGFloat = 760
        static let readingLineSpacing: CGFloat = 2
    }

    /// Shared horizontal/vertical padding tokens for main-window pages.
    /// Prefer these over ad-hoc literals so split panes + scroll pages stay visually aligned.
    enum PageInsets {
        static let rail: CGFloat = 12
        static let detail: CGFloat = 20
        static let sectionGap: CGFloat = 12
    }

    /// Preferred `NavigationSplitView` column widths so split resizing doesn't fight the window caps.
    enum SplitColumns {
        /// Root app chrome sidebar (Home/History/…).
        static let mainSidebarMin: CGFloat = 190
        static let mainSidebarIdeal: CGFloat = 220
        static let mainSidebarMax: CGFloat = 230

        /// Narrow “rail” lists (Memory categories).
        static let railListMin: CGFloat = 170
        static let railListIdeal: CGFloat = 210
        static let railListMax: CGFloat = 260

        /// Primary list columns (History, Modes, Memory item lists).
        ///
        /// Bumped from 260 \u{2192} 300 in PR5: the History row packs an
        /// app icon, status dot, two-line title/secondary, action
        /// chips, a timestamp, and a Replay button. At 260pt the chip
        /// stack and timestamp crowded each other on narrow windows;
        /// 300pt gives the row room to render without wrapping.
        static let primaryListMin: CGFloat = 300
        static let primaryListIdeal: CGFloat = 340
        static let primaryListMax: CGFloat = 380

        /// Secondary list columns (split panes with list + editor). Also used for Snippets & Memory item list + detail (stable width vs nested `NavigationSplitView`).
        static let secondaryListMin: CGFloat = 240
        static let secondaryListIdeal: CGFloat = 280
        static let secondaryListMax: CGFloat = 340

        /// Memory “category rail” inside Snippets & Memory (Vocabulary/Snippets/…).
        static let memoryCategoryRailMin: CGFloat = 170
        static let memoryCategoryRailIdeal: CGFloat = 190
        static let memoryCategoryRailMax: CGFloat = 220
    }

    static func windowBackground(_ scheme: ColorScheme) -> Color {
        // Light mode adopts the calmer stone paper. Dark mode unchanged.
        scheme == .dark
            ? Color(red: 6/255, green: 8/255, blue: 13/255)
            : JunoUI.Calm.paper
    }

    static func cardBackground(_ scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(red: 22/255, green: 25/255, blue: 35/255)
            : JunoUI.Calm.cardLight
    }

    static func railBackground(_ scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(red: 16/255, green: 20/255, blue: 29/255)
            : Color(red: 239/255, green: 236/255, blue: 229/255).opacity(0.98)
    }

    static func stageBackground(_ scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(red: 10/255, green: 12/255, blue: 18/255)
            : Color(red: 247/255, green: 245/255, blue: 240/255).opacity(0.98)
    }

    static func elevatedCard(_ scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(red: 28/255, green: 32/255, blue: 44/255)
            : Color(red: 251/255, green: 250/255, blue: 246/255)
    }

    static func border(_ scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color.white.opacity(0.105)
            : Color.black.opacity(0.10)
    }

    static func subtleBorder(_ scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color.white.opacity(0.075)
            : Color.black.opacity(0.07)
    }

    static func primaryText(_ scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(red: 244/255, green: 245/255, blue: 248/255)
            : Color(red: 15/255, green: 23/255, blue: 38/255)
    }

    static func secondaryText(_ scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(red: 155/255, green: 161/255, blue: 178/255)
            : Color(red: 86/255, green: 91/255, blue: 104/255)
    }

    static func tertiaryText(_ scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(red: 112/255, green: 119/255, blue: 138/255)
            : Color(red: 112/255, green: 113/255, blue: 122/255)
    }

    /// Secondary copy on form sheets (e.g. Memory add) where `secondaryText` is too dim on dark chrome.
    static func sheetBodySecondary(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color.white.opacity(0.78) : Color.black.opacity(0.58)
    }

    /// Accent tint used for interactive / brand moments
    static let accent: Color = JunoDesignTokens.accent

    /// Divider line
    static func divider(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? JunoDesignTokens.border.opacity(0.7) : Color.black.opacity(0.08)
    }
}

// MARK: - Global GroupBox style

struct JunoBrandGroupBoxStyle: GroupBoxStyle {
    @Environment(\.colorScheme) private var scheme

    func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: JunoTheme.Density.itemGap) {
            configuration.label
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
            configuration.content
        }
        .padding(JunoTheme.Density.cardPadding)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(JunoTheme.border(scheme), lineWidth: 0.5)
        )
    }
}

// MARK: - Premium card style (more elevated)

struct JunoPremiumCardStyle: ViewModifier {
    @Environment(\.colorScheme) private var scheme

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(JunoTheme.cardBackground(scheme))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(JunoTheme.border(scheme).opacity(scheme == .dark ? 0.7 : 0.22), lineWidth: 0.7)
            )
    }
}

extension View {
    func premiumCard() -> some View { modifier(JunoPremiumCardStyle()) }
}

struct JunoSidebarRailStyle: ViewModifier {
    @Environment(\.colorScheme) private var scheme

    func body(content: Content) -> some View {
        content
            .background(
                ZStack {
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .fill(JunoTheme.railBackground(scheme))
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [
                                    Color.white.opacity(scheme == .dark ? 0.035 : 0.18),
                                    Color.clear,
                                    JunoDesignTokens.accent.opacity(scheme == .dark ? 0.08 : 0.02)
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                }
            )
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .strokeBorder(Color.white.opacity(scheme == .dark ? 0.10 : 0.24), lineWidth: 0.8)
            )
    }
}

struct JunoStageSurfaceStyle: ViewModifier {
    @Environment(\.colorScheme) private var scheme

    func body(content: Content) -> some View {
        content
            .background(
                ZStack {
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .fill(JunoTheme.stageBackground(scheme))
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [
                                    JunoDesignTokens.accent.opacity(scheme == .dark ? 0.05 : 0.025),
                                    Color.clear,
                                    Color.white.opacity(scheme == .dark ? 0.015 : 0.12)
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                }
            )
            .overlay(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .strokeBorder(Color.white.opacity(scheme == .dark ? 0.09 : 0.18), lineWidth: 0.8)
            )
            .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
    }
}

struct JunoSubpaneSurfaceStyle: ViewModifier {
    @Environment(\.colorScheme) private var scheme

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(JunoTheme.elevatedCard(scheme))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(JunoTheme.border(scheme).opacity(scheme == .dark ? 0.68 : 0.18), lineWidth: 0.6)
            )
    }
}

// MARK: - Window background modifier

struct JunoBrandWindowBackground: ViewModifier {
    @Environment(\.colorScheme) private var scheme

    func body(content: Content) -> some View {
        content
            // Default appearance is light (`JunoUserDefaults.appearancePreference`); tokens cover both schemes.
            .tint(JunoDesignTokens.accent)
            .background(JunoTheme.windowBackground(scheme).ignoresSafeArea())
    }
}

extension View {
    func junoBrandWindow() -> some View {
        modifier(JunoBrandWindowBackground())
    }

    func junoSidebarRail() -> some View {
        modifier(JunoSidebarRailStyle())
    }

    func junoStageSurface() -> some View {
        modifier(JunoStageSurfaceStyle())
    }

    func junoSubpaneSurface() -> some View {
        modifier(JunoSubpaneSurfaceStyle())
    }
}

// MARK: - Page scaffold (consistent padding + chrome)

struct JunoPageScaffoldModifier: ViewModifier {
    let horizontal: CGFloat
    let vertical: CGFloat

    func body(content: Content) -> some View {
        content
            .padding(.horizontal, horizontal)
            .padding(.vertical, vertical)
    }
}

extension View {
    /// Standard padding for split-pane “rail” columns (History/Modes/Memory list columns).
    func junoSplitPanePadding() -> some View {
        modifier(JunoPageScaffoldModifier(horizontal: JunoTheme.PageInsets.rail, vertical: JunoTheme.PageInsets.rail))
    }

    /// Standard padding for full-width scroll/detail pages (Settings/App rules/Home hero grids).
    func junoDetailPagePadding() -> some View {
        modifier(JunoPageScaffoldModifier(horizontal: JunoTheme.PageInsets.detail, vertical: JunoTheme.PageInsets.detail))
    }

    /// Caps a detail page's content at a comfortable reading width and centers
    /// it inside the window. Prevents Home/Settings from looking stretched when
    /// the user has resized the window wide enough for History's split layout.
    func junoCenteredReadingPane(maxWidth: CGFloat = 980) -> some View {
        HStack(spacing: 0) {
            Spacer(minLength: 0)
            self.frame(maxWidth: maxWidth)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Button styling helpers

struct JunoPrimaryActionButtonStyle: ButtonStyle {
    @Environment(\.colorScheme) private var scheme
    @Environment(\.isEnabled) private var isEnabled
    @Environment(\.controlSize) private var controlSize

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: fontSize, weight: .semibold, design: .rounded))
            .foregroundStyle(foreground)
            .padding(.horizontal, horizontalPadding)
            .frame(minHeight: minHeight)
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(fill(configuration.isPressed))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(stroke, lineWidth: 0.7)
            )
            .shadow(
                color: shadowColor,
                radius: isEnabled && !configuration.isPressed ? 8 : 0,
                y: isEnabled && !configuration.isPressed ? 3 : 0
            )
    }

    private func fill(_ isPressed: Bool) -> Color {
        guard isEnabled else {
            return scheme == .dark
                ? Color.white.opacity(0.10)
                : Color.black.opacity(0.08)
        }
        let base = scheme == .dark
            ? Color(red: 20/255, green: 38/255, blue: 61/255)
            : JunoDesignTokens.iconBg
        return base.opacity(isPressed ? 0.84 : 1)
    }

    private var foreground: Color {
        isEnabled
            ? .white
            : (scheme == .dark ? Color.white.opacity(0.42) : Color.black.opacity(0.36))
    }

    private var stroke: Color {
        guard isEnabled else {
            return scheme == .dark ? Color.white.opacity(0.08) : Color.black.opacity(0.06)
        }
        return scheme == .dark
            ? Color.white.opacity(0.16)
            : Color.black.opacity(0.10)
    }

    private var shadowColor: Color {
        Color.black.opacity(scheme == .dark ? 0.28 : 0.12)
    }

    private var minHeight: CGFloat {
        switch controlSize {
        case .mini: return 22
        case .small: return 26
        case .regular: return 32
        case .large: return 38
        case .extraLarge: return 44
        @unknown default: return 32
        }
    }

    private var horizontalPadding: CGFloat {
        switch controlSize {
        case .mini: return 8
        case .small: return 10
        case .regular: return 13
        case .large: return 16
        case .extraLarge: return 18
        @unknown default: return 13
        }
    }

    private var fontSize: CGFloat {
        switch controlSize {
        case .mini: return 10.5
        case .small: return 11.5
        case .regular: return 13
        case .large: return 14.5
        case .extraLarge: return 15.5
        @unknown default: return 13
        }
    }

    private var cornerRadius: CGFloat {
        switch controlSize {
        case .mini, .small: return 7
        case .regular: return 8
        case .large: return 10
        case .extraLarge: return 11
        @unknown default: return 8
        }
    }
}

/// Small destructive action button. Red text on a tinted-red transparent
/// background with a subtle red border. Pressed state slightly darker. Used
/// for inline "Clear" / "Delete" controls on the Privacy page so they read
/// as Juno (not generic `.bordered`) without escalating to a heavy filled
/// destructive primary.
/// Apple-style restrained destructive control. The full ``danger`` token
/// is system-red (1.0/0.23/0.18) and reads as an alert when used as a
/// button background — we save that intensity for confirmation dialogs
/// and severity indicators, not for inline "Clear" / "Delete" buttons
/// next to user data. This style mirrors Apple Settings' destructive
/// rows: muted text in a danger hue, no fill, a hairline border that
/// becomes a soft tint on hover/press for affordance.
struct JunoDestructiveSmallButtonStyle: ButtonStyle {
    let scheme: ColorScheme
    var isEnabled: Bool = true
    @State private var hovering: Bool = false

    // Muted danger — slightly desaturated, slightly darker than the
    // alert-red token. Reads as "destructive intent" without shouting.
    private var dangerTint: Color {
        Color(red: 0.78, green: 0.30, blue: 0.28)
    }

    func makeBody(configuration: Configuration) -> some View {
        let pressed = configuration.isPressed
        return configuration.label
            .font(.system(size: 12, weight: .semibold, design: .rounded))
            .foregroundStyle(isEnabled ? dangerTint : JunoTheme.tertiaryText(scheme))
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(
                        isEnabled
                            ? dangerTint.opacity(pressed ? 0.10 : (hovering ? 0.06 : 0.0))
                            : Color.clear
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(
                        isEnabled
                            ? dangerTint.opacity(hovering || pressed ? 0.35 : 0.20)
                            : JunoTheme.border(scheme).opacity(0.30),
                        lineWidth: 0.6
                    )
            )
            .opacity(isEnabled ? 1.0 : 0.55)
            .onHover { h in withAnimation(.easeOut(duration: 0.14)) { hovering = h } }
    }
}

extension View {
    /// Suppress the heavy blue keyboard-focus ring SwiftUI draws around
    /// our buttons. Keeps keyboard accessibility (Tab still moves focus)
    /// but kills the visual clutter. Applied to every Juno button helper
    /// so we never have to remember it per call site.
    @ViewBuilder
    func junoNoFocusRing() -> some View {
        if #available(macOS 14.0, *) {
            self.focusEffectDisabled()
        } else {
            self
        }
    }

    /// Primary actions (create/save/install).
    func junoPrimaryActionButton() -> some View {
        self.buttonStyle(JunoPrimaryActionButtonStyle()).controlSize(.small).junoNoFocusRing()
    }

    /// Secondary actions (refresh/open/dismiss).
    func junoSecondaryActionButton() -> some View {
        self.buttonStyle(.bordered).controlSize(.small).junoNoFocusRing()
    }

    /// Low-emphasis actions (textual / inline).
    func junoTertiaryActionButton() -> some View {
        self.buttonStyle(.borderless).controlSize(.small).junoNoFocusRing()
    }

    func junoPageCard(padding: CGFloat = JunoTheme.Density.cardPadding) -> some View {
        self
            .padding(padding)
            .premiumCard()
    }
}

// MARK: - NSVisualEffectView wrapper (for HUD blur)

struct VisualEffectBlur: NSViewRepresentable {
    var material: NSVisualEffectView.Material = .hudWindow
    var blendingMode: NSVisualEffectView.BlendingMode = .behindWindow
    var state: NSVisualEffectView.State = .active

    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = material
        v.blendingMode = blendingMode
        v.state = state
        return v
    }
    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.blendingMode = blendingMode
        nsView.state = state
    }
}
