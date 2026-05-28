import SwiftUI

/// Juno brand timing curves and durations.
///
/// CSS ease tokens →
///   --ease-spring:  cubic-bezier(0.34, 1.56, 0.64, 1)   Pill expand, wake, beat
///   --ease-snap:    cubic-bezier(0.40, 0.00, 0.20, 1)    Scan, draft draw
///   --ease-out:     cubic-bezier(0.00, 0.00, 0.20, 1)    Quick exits
enum JunoBrandKitMotion {

    // MARK: - Wake (behavior 02)
    // scale(0.92) in 80ms ease-in → scale(1.16→1.0) in 270ms spring

    /// First 80ms: mark compresses to 0.92 with ease-in
    static let wakeCompressDuration: Double = 0.08
    static var wakeCompress: Animation { .easeIn(duration: wakeCompressDuration) }

    /// Remaining 270ms: spring over-shoot from 0.92 to 1.16, settles at 1.0
    static var wakeExpand: Animation {
        .timingCurve(0.34, 1.56, 0.64, 1, duration: 0.27)
    }

    // MARK: - Word Beat (behavior 03)
    // scale(1.0 → 1.12 → 0.97 → 1.0) over 280ms total

    /// 140ms: up to 1.12 with material ease
    static var wordBeatIn: Animation { .timingCurve(0, 0, 0.2, 1, duration: 0.14) }
    /// 80ms: rebound to 0.97
    static var wordBeatRebound: Animation { .easeOut(duration: 0.08) }
    /// 80ms: settle at 1.0
    static var wordBeatSettle: Animation { .easeOut(duration: 0.08) }

    // MARK: - Breath bars (brand kit: 320ms transition)

    static let breathBarHeightSeconds: Double = 0.32
    static var breathBarHeight: Animation { .easeInOut(duration: breathBarHeightSeconds) }

    // MARK: - Scan shimmer (behavior 04)
    // 580ms left→right, ease-snap

    static let scanDuration: Double = 0.58
    static var scan: Animation { .timingCurve(0.4, 0, 0.2, 1, duration: scanDuration) }

    // MARK: - Draft flash (behavior 05)
    // 260ms stroke draw + 200ms opacity fade

    static let draftDrawDuration: Double = 0.26
    static let draftFadeDuration: Double = 0.20

    // MARK: - Error shake (behavior 06)
    // 320ms / 3 cycles: 0 → -3px → 3px → -2px → 2px → 0

    static let shakeDuration: Double = 0.32

    // MARK: - Processing dots

    static let processingDotCycle: Double  = 1.1
    static let processingDotStagger: Double = 0.18

    // MARK: - Done row / inner fade

    static let doneRowFadeSeconds: Double = 0.35
    static var doneRowAppear: Animation { .easeOut(duration: doneRowFadeSeconds) }

    static let islandInnerFadeSeconds: Double = 0.15
    static var islandInnerFade: Animation { .easeOut(duration: islandInnerFadeSeconds) }

    // MARK: - Word text slide-in

    static let wordInDuration: Double = 0.20
    static var wordIn: Animation { .easeOut(duration: wordInDuration) }

    // MARK: - Idle breathe (behavior 01)
    // scale(1.0 → 1.025 → 1.0) · 4.8s · ease-in-out · infinite

    static var idleBreathe: Animation {
        .easeInOut(duration: 4.8).repeatForever(autoreverses: true)
    }

    // MARK: - Comma beat in HUD (spring overshoot)

    static let commaBeatSeconds: Double = 0.22
    static var commaBeat: Animation {
        .timingCurve(0.34, 1.56, 0.64, 1, duration: commaBeatSeconds)
    }
}
