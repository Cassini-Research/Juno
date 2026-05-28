import AppKit
import Foundation

// Usage:
//   swift scripts/generate_juno_app_icon_icns.swift /path/to/AppIcon.icns
//
// Generates the Juno app icon:
// - rounded square, radius 22.5%
// - background #0c1428
// - white Juno comma mark (SVG viewBox 60×84)

let args = CommandLine.arguments
guard args.count >= 2 else {
    fputs("usage: swift generate_juno_app_icon_icns.swift /path/to/AppIcon.icns\n", stderr)
    exit(2)
}

let outURL = URL(fileURLWithPath: args[1])
let fm = FileManager.default

let iconsetDir = outURL.deletingLastPathComponent().appendingPathComponent("AppIcon.iconset")
try? fm.removeItem(at: iconsetDir)
try fm.createDirectory(at: iconsetDir, withIntermediateDirectories: true)

let bg = NSColor(calibratedRed: 12 / 255, green: 20 / 255, blue: 40 / 255, alpha: 1) // #0c1428

func drawComma(in rect: CGRect, ctx: CGContext) {
    let vbW: CGFloat = 60
    let vbH: CGFloat = 84
    let scale = min(rect.width / vbW, rect.height / vbH)

    ctx.saveGState()
    defer { ctx.restoreGState() }

    // Flip Y to match SVG/SwiftUI coordinate sense.
    ctx.translateBy(x: rect.minX, y: rect.minY + rect.height)
    ctx.scaleBy(x: scale, y: -scale)
    ctx.translateBy(x: (rect.width / scale - vbW) / 2, y: (rect.height / scale - vbH) / 2)

    ctx.setFillColor(NSColor.white.cgColor)
    ctx.fillEllipse(in: CGRect(x: 30 - 22, y: 28 - 22, width: 44, height: 44))

    let tail = CGMutablePath()
    tail.move(to: CGPoint(x: 21, y: 46))
    tail.addCurve(to: CGPoint(x: 12, y: 78), control1: CGPoint(x: 14, y: 58), control2: CGPoint(x: 10, y: 70))
    tail.addCurve(to: CGPoint(x: 29, y: 78), control1: CGPoint(x: 14, y: 84), control2: CGPoint(x: 24, y: 85))
    tail.addCurve(to: CGPoint(x: 27, y: 47), control1: CGPoint(x: 34, y: 71), control2: CGPoint(x: 31, y: 58))
    tail.closeSubpath()

    ctx.addPath(tail)
    ctx.fillPath()
}

func render(size: Int) -> NSImage {
    let side = CGFloat(size)
    let img = NSImage(size: NSSize(width: side, height: side))
    img.lockFocus()
    defer { img.unlockFocus() }

    let rect = CGRect(x: 0, y: 0, width: side, height: side)
    let r = side * 0.225
    let shell = NSBezierPath(roundedRect: rect, xRadius: r, yRadius: r)
    bg.setFill()
    shell.fill()

    guard let ctx = NSGraphicsContext.current?.cgContext else { return img }
    let insetRect = rect.insetBy(dx: side * 0.18, dy: side * 0.14)
    drawComma(in: insetRect, ctx: ctx)
    return img
}

func writePNG(_ img: NSImage, to url: URL) throws {
    guard let tiff = img.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff),
          let png = rep.representation(using: .png, properties: [:])
    else {
        throw NSError(domain: "JunoIcon", code: 1)
    }
    try png.write(to: url)
}

let sizes: [(name: String, px: Int)] = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

for s in sizes {
    let img = render(size: s.px)
    try writePNG(img, to: iconsetDir.appendingPathComponent(s.name))
}

func appendBE32(_ value: UInt32, to data: inout Data) {
    data.append(UInt8((value >> 24) & 0xff))
    data.append(UInt8((value >> 16) & 0xff))
    data.append(UInt8((value >> 8) & 0xff))
    data.append(UInt8(value & 0xff))
}

func appendFourCC(_ raw: String, to data: inout Data) {
    data.append(raw.data(using: .ascii)!)
}

func writeFallbackICNS() throws {
    let chunks: [(String, String)] = [
        ("icp4", "icon_16x16.png"),
        ("icp5", "icon_32x32.png"),
        ("icp6", "icon_32x32@2x.png"),
        ("ic07", "icon_128x128.png"),
        ("ic08", "icon_128x128@2x.png"),
        ("ic09", "icon_256x256@2x.png"),
        ("ic10", "icon_512x512@2x.png"),
    ]
    var body = Data()
    for (type, name) in chunks {
        let png = try Data(contentsOf: iconsetDir.appendingPathComponent(name))
        appendFourCC(type, to: &body)
        appendBE32(UInt32(png.count + 8), to: &body)
        body.append(png)
    }
    var out = Data()
    appendFourCC("icns", to: &out)
    appendBE32(UInt32(body.count + 8), to: &out)
    out.append(body)
    try out.write(to: outURL)
}

let task = Process()
task.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
task.arguments = ["-c", "icns", iconsetDir.path, "-o", outURL.path]
try task.run()
task.waitUntilExit()

if task.terminationStatus != 0 {
    fputs("iconutil failed: \(task.terminationStatus); writing fallback ICNS\n", stderr)
    try writeFallbackICNS()
}

try? fm.removeItem(at: iconsetDir)
print("Wrote \(outURL.path)")
