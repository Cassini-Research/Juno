import AppKit
import SwiftUI

/// Raster Juno comma for **Dock** and **menu bar** where SwiftUI `Canvas` in
/// `MenuBarExtra` labels can fail to appear. Geometry matches `JunoCommaMark`.
enum JunoBrandMarkRasterizer {
    private static var installedDockIcon = false

    /// Same RGB as the light-mode `JunoDesignTokens.accent` for menu-bar tint parity.
    static let accentNSColor = NSColor(red: 12 / 255, green: 20 / 255, blue: 40 / 255, alpha: 1)
    static let mutedNSColor = NSColor(red: 74 / 255, green: 71 / 255, blue: 96 / 255, alpha: 1)
    private static let iconBg = NSColor(red: 12 / 255, green: 20 / 255, blue: 40 / 255, alpha: 1) // #0c1428

    /// Sets `NSApplication.shared.applicationIconImage` once (SwiftPM binary has no `.icns`).
    static func installDockIconIfNeeded() {
        guard !installedDockIcon else { return }
        installedDockIcon = true
        let side: CGFloat = 512
        let img = NSImage(size: NSSize(width: side, height: side), flipped: false) { rect in
            drawAppIcon(in: rect)
            return true
        }
        img.isTemplate = false
        NSApplication.shared.applicationIconImage = img
    }

    /// Template mark for the system menu bar (adapts to light/dark bar).
    static func menuBarTemplateImage() -> NSImage {
        let s: CGFloat = 22
        let img = NSImage(size: NSSize(width: s, height: s), flipped: false) { rect in
            // Template symbols are typically drawn in black; the system tints them.
            drawCommaFill(in: rect, fill: .black)
            return true
        }
        img.isTemplate = true
        return img
    }

    /// Non-template accent mark when dictation is active (listening / refining).
    static func menuBarTintedImage(accent: NSColor) -> NSImage {
        let s: CGFloat = 22
        let img = NSImage(size: NSSize(width: s, height: s), flipped: false) { rect in
            drawCommaFill(in: rect, fill: accent)
            return true
        }
        img.isTemplate = false
        return img
    }

    private static func drawAppIcon(in rect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        ctx.saveGState()
        defer { ctx.restoreGState() }

        // Brand kit app icon shell: 22.5% rounded square.
        let r = min(rect.width, rect.height) * 0.225
        let shell = NSBezierPath(roundedRect: rect, xRadius: r, yRadius: r)
        iconBg.setFill()
        shell.fill()

        // White comma mark centered, matching the kit’s proportions.
        drawCommaFill(in: rect.insetBy(dx: rect.width * 0.18, dy: rect.height * 0.14), fill: .white)
    }

    /// Draw the SVG-viewBox comma into an AppKit `CGContext` without inverting it.
    private static func drawCommaFill(in rect: NSRect, fill: NSColor) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        ctx.saveGState()
        defer { ctx.restoreGState() }

        let vbW: CGFloat = 64
        let vbH: CGFloat = 92
        let scale = min(rect.width / vbW, rect.height / vbH)

        // Align to rect and flip Y to match the SVG/SwiftUI coordinate system.
        ctx.translateBy(x: rect.minX, y: rect.minY + rect.height)
        ctx.scaleBy(x: scale, y: -scale)
        ctx.translateBy(x: (rect.width / scale - vbW) / 2, y: (rect.height / scale - vbH) / 2)

        fill.setFill()
        ctx.fillEllipse(in: JunoCommaMark.headDiskRect)
        ctx.addPath(JunoCommaMark.tailPath().cgPath)
        ctx.fillPath()
    }
}
