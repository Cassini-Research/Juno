import SwiftUI

/// 7-bar waveform presets.
/// BP = [[2,5,4,10,5,7,3],[3,8,6,14,8,10,4],[2,4,8,11,6,5,2],[1,3,4,8,3,6,2],[4,9,7,13,9,8,4]]
/// Bar: 2px wide · gap: 2.5px · height transition: 320ms ease
struct JunoBreathBars: View {
    let active: Bool
    let rms: Float

    private static let multipliers: [CGFloat] = [0.55, 0.75, 0.95, 1.15, 0.95, 0.75, 0.55]

    var body: some View {
        let level = CGFloat(min(max(rms * 28, 0), 1))
        let base: CGFloat = active ? 2 : 1
        let peak: CGFloat = active ? 14 : 4
        let heights: [CGFloat] = Self.multipliers.map { m in
            base + (peak * level * m)
        }
        HStack(alignment: .center, spacing: 2.5) {
            ForEach(0..<7, id: \.self) { i in
                Capsule()
                    .fill(Color.white.opacity(active ? 0.80 : 0.25))
                    .frame(width: 2, height: max(1, heights[i]))
                    .animation(JunoBrandKitMotion.breathBarHeight, value: heights[i])
            }
        }
        .frame(height: 18)
    }
}
