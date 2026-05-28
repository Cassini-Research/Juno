import SwiftUI

/// Full-screen brand signature moment.
struct JunoBrandMilestoneOverlay: View {
    let variant: JunoMilestoneNotifier.Variant

    @State private var commaScale: CGFloat = 1
    @State private var showCaption: Bool = false
    @State private var showWordmark: Bool = false

    var body: some View {
        // Sized to fit *inside* the HUD island shell (440pt wide × ~88pt tall)
        // rather than ignoring safe area and bleeding past the panel frame.
        // The 72pt comma was overflowing the HUD body, which read as a
        // glitchy oversized logo. We render a contained, in-place flourish
        // instead.
        ZStack {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(JunoDesignTokens.ink.opacity(0.92))
            HStack(spacing: 14) {
                ZStack {
                    JunoCommaMark(color: JunoDesignTokens.paper, scale: 0.55)
                        .frame(width: 32, height: 44)
                        .scaleEffect(commaScale)
                        .animation(JunoDesignTokens.beatSpring, value: commaScale)
                }
                if variant == .signatureFull || variant == .onboardingHero {
                    Text("Juno")
                        .font(.system(size: 22, weight: .semibold, design: .rounded))
                        .foregroundStyle(JunoDesignTokens.paper)
                        .opacity(showWordmark ? 1 : 0)
                        .offset(x: showWordmark ? 0 : 8)
                        .animation(.easeOut(duration: 0.40), value: showWordmark)
                }
                Spacer(minLength: 0)
                if variant != .onboardingHero {
                    Text("100 WORDS")
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .tracking(1.6)
                        .foregroundStyle(JunoDesignTokens.paper.opacity(0.55))
                        .opacity(showCaption ? 1 : 0)
                }
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
        }
        .frame(width: JunoDesignTokens.islandWidth)
        .clipped()
        .onAppear {
            commaScale = 1.12
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
                commaScale = 1
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                showCaption = true
            }
            if variant == .signatureFull || variant == .onboardingHero {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.55) {
                    showWordmark = true
                }
            }
        }
    }
}
