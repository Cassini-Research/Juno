import SwiftUI

// MARK: - Shared HUD island primitives
//
// These are the visual primitives shared between the full HUD
// (`JunoBrandIslandStack`) and the compact HUD (`JunoBrandIslandCompact`).
// The brand kit (April 2026) only specifies one capsule shell + one set of
// motion tokens for the dictation island — both HUD modes must render against
// the same shell and same motion vocabulary, so each lives in this file as an
// internal type rather than being duplicated per HUD.
//
// Anything in this file is also reachable by tests / future surfaces; do not
// add HUD-mode-specific styling here. State logic and layout stay in the
// per-mode files.

/// Single matte capsule used by both the full island and the compact pill —
/// blur + dark ink + optional danger wash + hairline. Keeps the dictation HUD
/// reading as the *same surface* across modes.
struct JunoIslandBackground: View {
    let danger: Bool

    var body: some View {
        ZStack {
            VisualEffectBlur(material: .hudWindow, blendingMode: .behindWindow)
                .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(Color(red: 12/255, green: 10/255, blue: 20/255).opacity(0.92))
            if danger {
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(JunoDesignTokens.danger.opacity(0.28))
            }
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .strokeBorder(Color.white.opacity(0.10), lineWidth: 0.5)
        }
    }
}

// MARK: - Processing dots (brand kit @keyframes pdA)

/// 3-dot ellipsis with the brand kit `pdA` keyframes (1.1s cycle, 0.18s stagger).
struct ProcessingDots: View {
    private static func pdScale(phase: Double) -> CGFloat {
        let c = JunoBrandKitMotion.processingDotCycle
        let p = phase.truncatingRemainder(dividingBy: c) / c
        if p < 0.4 { return CGFloat(p / 0.4) }
        if p < 0.8 { return CGFloat(1 - (p - 0.4) / 0.4) }
        return 0
    }

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30)) { t in
            let base = t.date.timeIntervalSinceReferenceDate
            HStack(spacing: 5) {
                ForEach(0..<3, id: \.self) { i in
                    let s = Self.pdScale(phase: base + Double(i) * JunoBrandKitMotion.processingDotStagger)
                    Circle()
                        .fill(Color.white)
                        .frame(width: max(0.5, 6 * s), height: max(0.5, 6 * s))
                }
            }
        }
    }
}

// MARK: - Scan shimmer (behavior 04 — 580ms ease-snap)

/// Vertical bar that sweeps left → right over the comma during refining.
struct JunoScanShimmer: View {
    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30)) { t in
            let phase = abs(sin(t.date.timeIntervalSinceReferenceDate * (1 / JunoBrandKitMotion.scanDuration)))
            GeometryReader { g in
                Rectangle()
                    .fill(Color.white.opacity(0.82))
                    .frame(width: 2.5, height: g.size.height)
                    .offset(x: phase * (g.size.width - 2.5))
            }
        }
    }
}
