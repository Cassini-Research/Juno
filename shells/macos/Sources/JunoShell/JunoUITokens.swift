import SwiftUI

// MARK: - JunoUI — semantic UI tokens layered atop JunoDesignTokens
//
// `JunoDesignTokens` is the brand kit (HUD pixels + motion). `JunoTheme`
// wires schemes to surface roles. This file adds the *semantic* layer the
// rest of the app should reach for first: a typographic ladder, a 4-pt
// spacing rhythm, three hairline tiers, and a small pool of UI-motion
// curves that sit above `JunoBrandKitMotion` (which is reserved for HUD
// brand moments). Everything here is additive — existing surfaces keep
// working unchanged until a polish pass migrates them.

enum JunoUI {

    // MARK: Spacing — 4-pt scale, named, no intermediates
    enum Spacing {
        static let xxs: CGFloat = 2
        static let xs:  CGFloat = 4
        static let s:   CGFloat = 8
        static let m:   CGFloat = 12
        static let l:   CGFloat = 18
        static let xl:  CGFloat = 28
        static let xxl: CGFloat = 44
    }

    // MARK: Type ladder — semantic, paired with weight + design
    //
    // Use these instead of `.font(.system(size: …))` literals. The whole
    // app should be expressible in 9 named styles; if a screen wants a
    // tenth, that's a signal we're inventing instead of reusing.
    enum TypeStyle {
        case display       // 30 / .semibold / .rounded — one per screen
        case title         // 22 / .semibold / .rounded — section heads
        case subtitle      // 17 / .medium   / .rounded — card titles
        case body          // 14 / .regular  / .rounded — default reading
        case bodyEmphasis  // 14 / .medium   / .rounded — inline emphasis
        case label         // 12 / .medium   / .rounded — field labels
        case caption       // 11 / .regular  / .rounded — footnotes
        case eyebrow       // 10 / .semibold / .monospaced + tracking — section eyebrows
        case mono          // 12 / .regular  / .monospaced — diagnostic / tabular

        var font: Font {
            switch self {
            case .display:      return .system(size: 30, weight: .semibold, design: .rounded)
            case .title:        return .system(size: 22, weight: .semibold, design: .rounded)
            case .subtitle:     return .system(size: 17, weight: .medium,   design: .rounded)
            case .body:         return .system(size: 14, weight: .regular,  design: .rounded)
            case .bodyEmphasis: return .system(size: 14, weight: .medium,   design: .rounded)
            case .label:        return .system(size: 12, weight: .medium,   design: .rounded)
            case .caption:      return .system(size: 11, weight: .regular,  design: .rounded)
            case .eyebrow:      return .system(size: 10, weight: .semibold, design: .monospaced)
            case .mono:         return .system(size: 12, weight: .regular,  design: .monospaced)
            }
        }

        var tracking: CGFloat {
            self == .eyebrow ? 1.4 : 0
        }
    }

    // MARK: Hairline opacity tiers — three only.
    //
    // Replace literal `.opacity(0.07/0.10/0.22/0.32/0.55…)` border calls
    // with `.faint`, `.regular`, `.strong`. The map is scheme-aware so
    // dark mode reads slightly more present than light without forcing
    // every site to think about it.
    enum HairlineTier { case faint, regular, strong }

    static func hairline(_ tier: HairlineTier, scheme: ColorScheme) -> Color {
        let dark = scheme == .dark
        switch tier {
        case .faint:   return (dark ? Color.white : Color.black).opacity(dark ? 0.08 : 0.07)
        case .regular: return (dark ? Color.white : Color.black).opacity(dark ? 0.14 : 0.11)
        case .strong:  return (dark ? Color.white : Color.black).opacity(dark ? 0.24 : 0.20)
        }
    }

    // MARK: UI motion — above the HUD, below the brand-kit moments.
    //
    // Three curves cover ~95% of UI transitions. `JunoBrandKitMotion`
    // remains the source of truth for HUD/brand beats; pages should reach
    // for these.
    enum Motion {
        /// Page swap / sheet present.
        static let pageSwap = Animation.easeOut(duration: 0.32)
        /// Card / panel reveal — soft spring with no overshoot kick.
        static let cardReveal = Animation.spring(response: 0.42, dampingFraction: 0.86)
        /// Button press / chip selection / tiny state changes.
        static let microPress = Animation.spring(response: 0.22, dampingFraction: 0.78)
        /// Hover/focus dim. Symmetric in/out.
        static let dim = Animation.easeInOut(duration: 0.18)
    }

    // MARK: Calmer palette.
    //
    // This palette is cool, soft, and restrained: a neutral stone for paper,
    // near-black with a hint of indigo for ink, and a deep-navy accent that
    // reads sophisticated rather than playful.
    // Existing surfaces keep `JunoDesignTokens.paper / .ink / .accent`;
    // new redesigned surfaces reach for these.
    enum Calm {
        /// #F2F0EB — cool stone paper, less yellow than `JunoDesignTokens.paper` (#f4f1ea).
        static let paper     = Color(red: 242/255, green: 240/255, blue: 235/255)
        /// #06080D — deep neutral base.
        static let ink       = Color(red:   6/255, green:   8/255, blue:  13/255)
        /// #FAF9F5 — elevated card on light stone.
        static let cardLight = Color(red: 250/255, green: 249/255, blue: 245/255)
        /// #1C202C — elevated card on cool ink.
        static let cardDark  = Color(red:  28/255, green:  32/255, blue:  44/255)
        /// Deep navy accent. Used for selected / active / interactive.
        static let accent    = JunoDesignTokens.accent
        /// #B59A77 — warm sand, reserved for one-off delight (milestones,
        /// onboarding hero).
        static let highlight = Color(red: 181/255, green: 154/255, blue: 119/255)
        /// #5E5645 — deep ink-on-paper text tier 2.
        static let inkSoft   = Color(red:  94/255, green:  86/255, blue:  69/255)
        /// #348A6E — calm meadow green for "saved time" / success captions.
        /// Matches `JunoDesignTokens.meadow` but lives here so Calm-skinned
        /// surfaces can reach for it without an extra import path.
        static let meadow    = JunoDesignTokens.meadow
        /// #A06B2A — warm amber, used by failure rows on Home Recents.
        /// Less alarming than `JunoDesignTokens.danger`; signals "needs
        /// attention" without screaming.
        static let amber     = Color(red: 160/255, green: 107/255, blue:  42/255)
        /// Bar fill for non-emphasized days in the 7-day stats chart.
        /// Scheme-neutral on purpose — the today bar is `accent`.
        static func barRest(scheme: ColorScheme) -> Color {
            scheme == .dark ? Color.white.opacity(0.10) : Color.black.opacity(0.10)
        }
    }
}

// MARK: - View modifiers — let any view consume tokens fluently

extension View {
    /// Apply a semantic type style. Replaces `.font(.system(size:…))` literals.
    func junoType(_ style: JunoUI.TypeStyle) -> some View {
        let tracking = style.tracking
        return self
            .font(style.font)
            .tracking(tracking)
    }

    /// 1-pt hairline border at one of three opacity tiers.
    func junoHairlineBorder(
        _ tier: JunoUI.HairlineTier = .regular,
        cornerRadius: CGFloat = 0
    ) -> some View {
        modifier(JunoHairlineBorderModifier(tier: tier, cornerRadius: cornerRadius))
    }

}

/// 1-pt hairline rule. Use as a Divider replacement when you want the
/// opacity tier to be explicit and scheme-aware.
struct JunoHairlineRule: View {
    @Environment(\.colorScheme) private var scheme
    let tier: JunoUI.HairlineTier

    init(_ tier: JunoUI.HairlineTier = .regular) { self.tier = tier }

    var body: some View {
        Rectangle()
            .fill(JunoUI.hairline(tier, scheme: scheme))
            .frame(height: 1)
    }
}

private struct JunoHairlineBorderModifier: ViewModifier {
    @Environment(\.colorScheme) private var scheme
    let tier: JunoUI.HairlineTier
    let cornerRadius: CGFloat

    func body(content: Content) -> some View {
        content.overlay(
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .strokeBorder(JunoUI.hairline(tier, scheme: scheme), lineWidth: 1)
        )
    }
}

// MARK: - Eyebrow label
//
// A small uppercased monospaced label used to title sections, steps, and
// status groups. Replaces the inline pattern that was repeated across
// JunoModesView / JunoActionsPage / JunoOnboarding.
struct JunoEyebrow: View {
    @Environment(\.colorScheme) private var scheme
    let text: String
    var color: Color? = nil

    var body: some View {
        Text(text.uppercased())
            .junoType(.eyebrow)
            .foregroundStyle(color ?? JunoTheme.secondaryText(scheme).opacity(scheme == .dark ? 0.92 : 0.78))
    }
}

// MARK: - Section block — eyebrow + content with consistent rhythm

struct JunoSection<Content: View>: View {
    let eyebrow: String?
    let title: String?
    @ViewBuilder var content: () -> Content

    init(
        eyebrow: String? = nil,
        title: String? = nil,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.eyebrow = eyebrow
        self.title = title
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: JunoUI.Spacing.s) {
            if let eyebrow {
                JunoEyebrow(text: eyebrow)
            }
            if let title {
                Text(title).junoType(.title)
            }
            content()
        }
    }
}

// MARK: - Premium product page primitives

struct JunoPageHeader<Trailing: View>: View {
    let eyebrow: String?
    let title: String
    let subtitle: String?
    @ViewBuilder var trailing: () -> Trailing

    @Environment(\.colorScheme) private var scheme

    init(
        eyebrow: String? = nil,
        title: String,
        subtitle: String? = nil,
        @ViewBuilder trailing: @escaping () -> Trailing
    ) {
        self.eyebrow = eyebrow
        self.title = title
        self.subtitle = subtitle
        self.trailing = trailing
    }

    var body: some View {
        HStack(alignment: .top, spacing: JunoUI.Spacing.l) {
            VStack(alignment: .leading, spacing: JunoUI.Spacing.xs) {
                if let eyebrow {
                    JunoEyebrow(text: eyebrow)
                }
                Text(title)
                    .junoType(.title)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .junoType(.body)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
            trailing()
        }
    }
}

extension JunoPageHeader where Trailing == EmptyView {
    init(eyebrow: String? = nil, title: String, subtitle: String? = nil) {
        self.eyebrow = eyebrow
        self.title = title
        self.subtitle = subtitle
        self.trailing = { EmptyView() }
    }
}

struct JunoPreferenceSection<Content: View>: View {
    let title: String
    let subtitle: String?
    @ViewBuilder var content: () -> Content

    @Environment(\.colorScheme) private var scheme

    init(
        title: String,
        subtitle: String? = nil,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: JunoUI.Spacing.m) {
            VStack(alignment: .leading, spacing: JunoUI.Spacing.xs) {
                JunoSectionLabel(text: title)
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .junoType(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            content()
        }
        .junoPageCard()
    }
}

struct JunoPreferenceRow<Trailing: View>: View {
    let title: String
    let subtitle: String?
    @ViewBuilder var trailing: () -> Trailing

    @Environment(\.colorScheme) private var scheme

    init(
        title: String,
        subtitle: String? = nil,
        @ViewBuilder trailing: @escaping () -> Trailing
    ) {
        self.title = title
        self.subtitle = subtitle
        self.trailing = trailing
    }

    var body: some View {
        HStack(alignment: subtitle == nil ? .center : .top, spacing: JunoUI.Spacing.m) {
            VStack(alignment: .leading, spacing: JunoUI.Spacing.xs) {
                Text(title)
                    .junoType(.bodyEmphasis)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .junoType(.caption)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: JunoUI.Spacing.m)
            trailing()
        }
    }
}

struct JunoQuietDisclosure<Content: View>: View {
    let title: String
    @Binding var isExpanded: Bool
    @ViewBuilder var content: () -> Content

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
            VStack(alignment: .leading, spacing: JunoUI.Spacing.m) {
                content()
            }
            .padding(.top, JunoUI.Spacing.s)
        } label: {
            Text(title)
                .junoType(.label)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
        }
    }
}
