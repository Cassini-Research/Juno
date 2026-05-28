import AppKit
import SwiftUI

/// Juno design tokens.
/// CSS variables → Swift constants. Typography uses system fonts that approximate
/// DM Sans / DM Mono / Space Grotesk; bundle licensed webfonts later for pixel-parity.
enum JunoDesignTokens {

    private static func adaptiveColor(light: NSColor, dark: NSColor) -> Color {
        let color = NSColor(name: nil) { appearance in
            let resolved = appearance.bestMatch(from: [.darkAqua, .aqua])
            return resolved == .darkAqua ? dark : light
        }
        return Color(nsColor: color)
    }

    // MARK: - Core Colors (CSS :root)

    /// #09080e — near-black background
    static let ink = Color(red:   9/255, green:   8/255, blue:  14/255)
    /// #f4f1ea — warm off-white background (light mode)
    static let paper = Color(red: 244/255, green: 241/255, blue: 234/255)
    /// #0f0e16 — elevated surface card (dark mode)
    static let card = Color(red:  15/255, green:  14/255, blue:  22/255)
    /// #1c1a28 — borders, dividers
    static let border = Color(red:  28/255, green:  26/255, blue:  40/255)
    /// Muted / secondary text. Dark mode deliberately lifts this above the
    /// original purple-gray so small labels remain comfortably legible.
    static let muted = adaptiveColor(
        light: NSColor(calibratedRed: 74/255, green: 71/255, blue: 96/255, alpha: 1),
        dark: NSColor(calibratedRed: 150/255, green: 156/255, blue: 174/255, alpha: 1)
    )
    /// Dim supporting chrome.
    static let dim = adaptiveColor(
        light: NSColor(calibratedRed: 40/255, green: 37/255, blue: 58/255, alpha: 1),
        dark: NSColor(calibratedRed: 77/255, green: 83/255, blue: 101/255, alpha: 1)
    )

    /// #0c1428 — app icon background and primary light-mode brand navy.
    static let iconBg = Color(red:  12/255, green:  20/255, blue:  40/255)

    /// Accent tint for brand/interactive UI.
    ///
    /// Light mode uses the same dock navy as the Dictate pill and app icon so
    /// Juno feels closer to native macOS chrome than a generic blue AI app.
    /// Dark mode lifts to a tempered steel-blue for legibility on ink surfaces
    /// without turning every control into bright default-blue chrome.
    static let accent = adaptiveColor(
        light: NSColor(calibratedRed: 12/255, green: 20/255, blue: 40/255, alpha: 1),
        dark: NSColor(calibratedRed: 126/255, green: 147/255, blue: 184/255, alpha: 1)
    )
    /// Accent at 15% opacity — subtle tint fills.
    static let accentDim = accent.opacity(0.15)

    /// Danger / error red
    static let danger = Color(red: 1.0, green: 0.23, blue: 0.18)

    /// Granted / success surfaces (onboarding, settings) — calmer than system `green`
    static let meadow = Color(red: 52/255, green: 138/255, blue: 110/255)

    // MARK: - Dictation island sizes (floating bar at top of screen)

    /// Dormant: brand kit 32×8 — a barely-visible thin pill. Used for transition.
    static let dormantSize    = CGSize(width: 36,  height: 10)
    /// Listening: minimum size — single-line pill the HUD opens at before the
    /// user has spoken anything. The HUD grows to fit content from here, capped
    /// by `listeningMaxHeight` so it never dominates the screen.
    static let listeningSize  = CGSize(width: 360, height: 44)
    /// Width when the live transcript starts wrapping. The HUD never renders wider
    /// than this so the pill stays comfortable to read.
    static let listeningMaxWidth: CGFloat = 460
    /// Maximum height the listening pill is allowed to grow to (≈ 3–4 lines of
    /// transcript + the status row). Beyond this, the inner scroll view scrolls.
    static let listeningMaxHeight: CGFloat = 160
    /// Max height of the live transcript scroll area inside the listening pill
    /// (≈ 3–4 lines of the transcript font).
    static let listeningTranscriptScrollMaxHeight: CGFloat = 92
    /// Fixed width of the unified HUD shell across listening / refining / copy-ready / done.
    /// Tighter than the legacy `listeningMaxWidth` so the pill reads as one capsule
    /// regardless of body content.
    static let islandWidth: CGFloat = 440
    /// Max height of the copy-ready transcript scroll area.
    static let copyReadyTranscriptMaxHeight: CGFloat = 140
    /// Processing: comma + dots + label (+ optional timer)
    static let processingSize = CGSize(width: 240, height: 50)
    /// Copy-ready idle state (full transcript scroll)
    static let copyReadyIslandSize = CGSize(width: 420, height: 200)
    /// Done: comma + "Text placed" + word count
    static let doneSize       = CGSize(width: 218, height: 46)

    // MARK: - Motion

    /// Pill size / show spring — spring(response:0.55, dampingFraction:0.78)
    static let pillSpring = Animation.spring(response: 0.55, dampingFraction: 0.78)
    /// Word beat punch-in — spring(response:0.28, dampingFraction:0.72)
    static let beatSpring = Animation.spring(response: 0.28, dampingFraction: 0.72)
    /// Inner fade-out duration (ms → s)
    static let innerFadeSeconds: Double = 0.18

    /// Breath bars tick interval  (brand kit: 320ms)
    static let breathInterval: TimeInterval = 0.32
    /// Word beat full duration
    static let wordBeatDuration: Double = 0.28
}
