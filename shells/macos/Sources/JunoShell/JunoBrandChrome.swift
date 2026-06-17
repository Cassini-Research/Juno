import SwiftUI

// MARK: - Brand header strip

/// Chrome header used above sidebars, in modals, and as the home hero.
struct JunoChromeBrandHeader: View {
    enum Style {
        /// Slim strip above the NavigationSplitView sidebar.
        case sidebar
        /// Large hero on the Home pane (animated mark + large wordmark).
        case hero
        /// Single-row for modals / settings panes.
        case compact
    }

    let style: Style
    var title: String = "Juno"
    var subtitle: String?
    /// Optional trailing context (e.g. current page name in the sidebar strip).
    var pageContext: String? = nil

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        switch style {
        case .sidebar:  sidebarHeader
        case .hero:     heroHeader
        case .compact:  compactHeader
        }
    }

    // MARK: Sidebar

    private var sidebarHeader: some View {
        HStack(spacing: 10) {
            JunoChromeAmbientMark(large: false)
            Text(title)
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
            Spacer(minLength: 0)
            if let pageContext, !pageContext.isEmpty {
                Text(pageContext)
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.85))
                    .tracking(0.15)
                    .transition(.opacity.combined(with: .move(edge: .trailing)))
                    .id(pageContext)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .animation(.easeInOut(duration: 0.18), value: pageContext)
    }

    // MARK: Hero

    private var heroHeader: some View {
        HStack(alignment: .center, spacing: 18) {
            JunoChromeAmbientMark(large: true)
            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.system(size: 24, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.system(size: 13, weight: .regular, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 6)
    }

    // MARK: Compact

    private var compactHeader: some View {
        HStack(spacing: 10) {
            JunoChromeAmbientMark(large: false)
            Text(title)
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
            Spacer(minLength: 0)
        }
        .padding(.vertical, 6)
    }
}

// MARK: - Ambient mark (idle breathe — behavior 01)

/// Brand mark for in-app chrome. Defaults to an editorial comma (no navy disc); use ``Presentation/dockDisc`` for Dock/menu parity.
struct JunoChromeAmbientMark: View {
    var large: Bool = false
    var presentation: Presentation = .editorial
    /// When `false`, skips `repeatForever` idle breathe. Keep this opt-in: the
    /// shell is a long-lived menu-bar app, and persistent SwiftUI repeat
    /// animations have shown up as main-thread layout churn during long idle.
    var idleBreathing: Bool = false
    @State private var breathe = false

    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// Scene phase observer. The app can spend hours inactive as a menu-bar
    /// utility, so repeat animations must only run while the scene is active.
    @Environment(\.scenePhase) private var scenePhase

    enum Presentation {
        /// Ink / light comma on paper-style surfaces (main window, settings headers).
        case editorial
        /// White comma for the navy home hero card.
        case onBrandDark
        /// Navy circle + white comma (matches the Dock icon asset).
        case dockDisc
    }

    private var circleSize: CGFloat { large ? 62 : 38 }
    private var markW: CGFloat { large ? 30 : 18 }
    private var markH: CGFloat { large ? 42 : 25 }
    private var shouldBreathe: Bool { idleBreathing && !reduceMotion && scenePhase == .active }

    var body: some View {
        Group {
            switch presentation {
            case .dockDisc:
                ZStack {
                    Circle()
                        .fill(JunoDesignTokens.iconBg)
                    JunoCommaMark(color: .white, scale: large ? 0.88 : 0.82)
                        .frame(width: markW, height: markH)
                }
            case .editorial:
                JunoCommaMark(color: JunoTheme.primaryText(scheme), scale: large ? 0.92 : 0.86)
                    .frame(width: markW * 1.06, height: markH * 1.06)
                    .shadow(color: Color.black.opacity(scheme == .dark ? 0.35 : 0.07), radius: 10, y: 4)
            case .onBrandDark:
                JunoCommaMark(color: Color.white, scale: large ? 0.94 : 0.88)
                    .frame(width: markW * 1.1, height: markH * 1.1)
                    .shadow(color: Color.black.opacity(0.22), radius: 8, y: 3)
            }
        }
        .frame(width: circleSize, height: circleSize)
        .scaleEffect(breathe ? 1.025 : 1.0)
        .onAppear {
            applyBreathingState(animated: false)
        }
        .onChange(of: idleBreathing) { _ in
            applyBreathingState(animated: true)
        }
        .onChange(of: reduceMotion) { _ in
            applyBreathingState(animated: true)
        }
        .onChange(of: scenePhase) { _ in
            applyBreathingState(animated: true)
        }
    }

    private func applyBreathingState(animated: Bool) {
        let next = shouldBreathe
        guard breathe != next else { return }
        if next, animated {
            withAnimation(JunoBrandKitMotion.idleBreathe) { breathe = true }
            return
        }
        var transaction = Transaction()
        transaction.disablesAnimations = true
        withTransaction(transaction) { breathe = next }
    }
}

// MARK: - Empty state

struct JunoChromeEmptyState: View {
    let title: String
    let message: String
    var symbol: String? = nil
    /// When false, avoids stretching to infinite height so the empty state can sit with sibling controls in a `VStack` without clipping.
    var expandsToFillAvailableSpace: Bool = true
    /// Lighter typography and icon — used e.g. Snippets & Memory list column in a narrow split.
    var compact: Bool = false

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let iconFont: CGFloat = compact ? 28 : 44
        let iconBubblePad: CGFloat = compact ? 14 : 20
        let titleFont: CGFloat = compact ? 15 : 18
        let messageFont: CGFloat = compact ? 12 : 13
        let outerPadding: CGFloat = expandsToFillAvailableSpace ? (compact ? 28 : 40) : (compact ? 18 : 24)
        let stackSpacing: CGFloat = compact ? 12 : 18

        VStack(spacing: stackSpacing) {
            if let symbol {
                Image(systemName: symbol)
                    .font(.system(size: iconFont, weight: .medium))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(JunoDesignTokens.accent.opacity(0.85))
                    .padding(iconBubblePad)
                    .background(
                        Circle()
                            .fill(JunoDesignTokens.accent.opacity(scheme == .dark ? 0.12 : 0.08))
                    )
            } else {
                JunoCommaMark(color: JunoDesignTokens.accent.opacity(0.45), scale: compact ? 0.95 : 1.1)
                    .frame(width: compact ? 40 : 48, height: compact ? 52 : 62)
            }
            VStack(spacing: compact ? 4 : 6) {
                Text(title)
                    .font(.system(size: titleFont, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text(message)
                    .font(.system(size: messageFont, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: compact ? 260 : 340)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(outerPadding)
        .frame(maxHeight: expandsToFillAvailableSpace ? .infinity : nil, alignment: .center)
    }
}

// MARK: - Split list column title (History / Modes)

/// Leading page title + trailing toolbar actions for the first column of a ``NavigationSplitView``.
struct JunoSplitColumnTitleRow<Trailing: View>: View {
    let title: String
    @ViewBuilder var trailing: () -> Trailing

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Text(title)
                .font(.system(size: 18, weight: .semibold, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
            Spacer(minLength: 0)
            trailing()
        }
    }
}

// MARK: - Section label style

struct JunoSectionLabel: View {
    let text: String
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Text(text.uppercased())
            .font(.system(size: 10, weight: .semibold, design: .monospaced))
            .tracking(1.2)
            .foregroundStyle(JunoTheme.secondaryText(scheme))
    }
}

// MARK: - Inline section chrome (in-content headers)

struct JunoInlineSearchField: View {
    var prompt: String
    @Binding var text: String
    var symbol: String = "magnifyingglass"
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: symbol)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
            TextField(prompt, text: $text)
                .textFieldStyle(.plain)
                .focusEffectDisabled()
                .font(.system(size: 13, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 11, style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 11, style: .continuous)
                .strokeBorder(JunoTheme.subtleBorder(scheme).opacity(0.85), lineWidth: 0.5)
        )
    }
}

struct JunoInlineSectionHeader<Leading: View, Trailing: View>: View {
    var title: String
    var subtitle: String? = nil
    var usePremiumCard: Bool = false
    @ViewBuilder var leading: () -> Leading
    @ViewBuilder var trailing: () -> Trailing

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let header = HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                if let subtitle, !subtitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text(subtitle)
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                }
            }
            Spacer(minLength: 0)
            leading()
            trailing()
        }
        .padding(usePremiumCard ? 14 : 0)

        Group {
            if usePremiumCard {
                header.premiumCard()
            } else {
                header
            }
        }
    }
}

// MARK: - Inline status banners (premium, consistent)

struct JunoInlineStatusBanner<Trailing: View>: View {
    enum Kind { case info, success, warning, danger }

    let kind: Kind
    let title: String
    var message: String? = nil
    var systemImage: String? = nil
    @ViewBuilder var trailing: () -> Trailing

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(accent)
                    .padding(.top, 1)
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(.subheadline, design: .rounded).weight(.semibold))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                if let message, !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text(message)
                        .font(.callout)
                        .foregroundStyle(JunoTheme.secondaryText(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
            trailing()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(fill)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(stroke, lineWidth: 0.5)
        )
    }

    private var accent: Color {
        switch kind {
        case .info:    return JunoDesignTokens.accent
        case .success: return JunoDesignTokens.meadow
        case .warning: return Color.orange
        case .danger:  return JunoDesignTokens.danger
        }
    }

    private var fill: Color {
        switch kind {
        case .info:    return JunoDesignTokens.accent.opacity(scheme == .dark ? 0.10 : 0.08)
        case .success: return JunoDesignTokens.meadow.opacity(0.10)
        case .warning: return Color.orange.opacity(0.10)
        case .danger:  return JunoDesignTokens.danger.opacity(0.10)
        }
    }

    private var stroke: Color {
        switch kind {
        case .info:    return JunoDesignTokens.accent.opacity(0.22)
        case .success: return JunoDesignTokens.meadow.opacity(0.22)
        case .warning: return Color.orange.opacity(0.22)
        case .danger:  return JunoDesignTokens.danger.opacity(0.22)
        }
    }
}

extension JunoInlineStatusBanner where Trailing == EmptyView {
    init(kind: Kind, title: String, message: String? = nil, systemImage: String? = nil) {
        self.kind = kind
        self.title = title
        self.message = message
        self.systemImage = systemImage
        self.trailing = { EmptyView() }
    }
}

// MARK: - List chrome (split panes)

struct JunoSplitListChrome: ViewModifier {
    func body(content: Content) -> some View {
        content
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
            .environment(\.defaultMinListRowHeight, JunoTheme.Density.listRowMinHeight)
    }
}

extension View {
    func junoSplitListChrome() -> some View {
        modifier(JunoSplitListChrome())
    }
}

// MARK: - Status badge

struct JunoStatusBadge: View {
    enum State { case ok, warning, error, neutral }
    let state: State
    let label: String

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(dotColor)
                .frame(width: 7, height: 7)
            Text(label)
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(textColor)
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 4)
        .background(
            Capsule().fill(dotColor.opacity(0.12))
        )
        .overlay(Capsule().strokeBorder(dotColor.opacity(0.25), lineWidth: 0.5))
    }

    private var dotColor: Color {
        switch state {
        case .ok:      return .green
        case .warning: return .orange
        case .error:   return JunoDesignTokens.danger
        case .neutral: return JunoDesignTokens.muted
        }
    }
    private var textColor: Color {
        switch state {
        case .ok:      return .green
        case .warning: return .orange
        case .error:   return JunoDesignTokens.danger
        case .neutral: return JunoDesignTokens.muted
        }
    }
}
