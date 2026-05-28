import SwiftUI

/// Juno comma mark geometry.
/// viewBox 0 0 64 92 · head: cx=32 cy=27 r=22 · tail: M19,46 C10,61 6,75 9,83 C12,89 23,90 30,83 C37,76 34,61 28,48 Z
struct JunoCommaMark: View {
    var color: Color = JunoDesignTokens.paper
    var scale: CGFloat = 1

    var body: some View {
        Canvas { context, size in
            let s = min(size.width / 64, size.height / 92) * scale
            let tx = (size.width  - 64 * s) / 2
            let ty = (size.height - 92 * s) / 2
            let t = CGAffineTransform(translationX: tx, y: ty).scaledBy(x: s, y: s)
            context.concatenate(t)
            var head = Path()
            head.addEllipse(in: Self.headDiskRect)
            context.fill(head, with: .color(color))
            context.fill(Self.tailPath(), with: .color(color))
        }
        .accessibilityHidden(true)
    }

    // MARK: - Shared geometry (used by rasterizer)

    /// Head disk in viewBox coordinates (64×92).
    /// cx=32, cy=27, r=22  →  origin=(10,5)
    static var headDiskRect: CGRect {
        CGRect(x: 10, y: 5, width: 44, height: 44)
    }

    /// Tail cubic spline in viewBox coordinates.
    /// SVG: M19,46 C10,61 6,75 9,83 C12,89 23,90 30,83 C37,76 34,61 28,48 Z
    static func tailPath() -> Path {
        var p = Path()
        p.move(to: CGPoint(x: 19, y: 46))
        p.addCurve(
            to:       CGPoint(x:  9, y: 83),
            control1: CGPoint(x: 10, y: 61),
            control2: CGPoint(x:  6, y: 75))
        p.addCurve(
            to:       CGPoint(x: 30, y: 83),
            control1: CGPoint(x: 12, y: 89),
            control2: CGPoint(x: 23, y: 90))
        p.addCurve(
            to:       CGPoint(x: 28, y: 48),
            control1: CGPoint(x: 37, y: 76),
            control2: CGPoint(x: 34, y: 61))
        p.closeSubpath()
        return p
    }
}
