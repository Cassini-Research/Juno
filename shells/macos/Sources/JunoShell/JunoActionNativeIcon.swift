import AppKit
import SwiftUI

struct JunoActionNativeIcon: View {
    let kind: JunoActionKind
    var size: CGFloat
    var fallbackColor: Color?

    var body: some View {
        if let image = Self.image(for: kind) {
            Image(nsImage: image)
                .resizable()
                .interpolation(.high)
                .scaledToFit()
                .frame(width: size, height: size)
                .accessibilityLabel(Text(kind.descriptor.displayName))
        } else {
            Image(systemName: kind.descriptor.symbolName)
                .font(.system(size: max(8, size * 0.48), weight: .semibold))
                .foregroundStyle(fallbackColor ?? kind.descriptor.accent)
                .frame(width: size, height: size)
                .accessibilityLabel(Text(kind.descriptor.displayName))
        }
    }

    private static func image(for kind: JunoActionKind) -> NSImage? {
        let bundleId = kind.descriptor.nativeBundleIdentifier
        guard let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleId) else {
            return nil
        }
        let image = NSWorkspace.shared.icon(forFile: url.path)
        image.size = NSSize(width: 128, height: 128)
        return image
    }
}

struct JunoActionNativeIconTile: View {
    let kind: JunoActionKind
    var tileSize: CGFloat
    var iconSize: CGFloat
    var fallbackTint: Color?
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: min(10, tileSize * 0.24), style: .continuous)
                .fill(JunoTheme.elevatedCard(scheme).opacity(scheme == .dark ? 0.78 : 0.92))
                .frame(width: tileSize, height: tileSize)
                .shadow(color: Color.black.opacity(scheme == .dark ? 0.24 : 0.08), radius: 2, x: 0, y: 1)
            JunoActionNativeIcon(
                kind: kind,
                size: iconSize,
                fallbackColor: fallbackTint ?? kind.descriptor.accent
            )
        }
        .frame(width: tileSize, height: tileSize)
    }
}

struct JunoAlarmInfoButton: View {
    @State private var isShowingInfo = false
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Button {
            isShowingInfo.toggle()
        } label: {
            Image(systemName: "info.circle.fill")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(JunoTheme.secondaryText(scheme).opacity(0.72))
                .frame(width: 18, height: 18)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .junoNoFocusRing()
        .help("How alarms work in Juno")
        .popover(isPresented: $isShowingInfo, arrowEdge: .top) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Alarm")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("Juno saves alarms as Calendar alerts so they can ring on time even if Juno is closed. Use Open Calendar to view, edit, or delete them.")
                    .font(.system(size: 12, weight: .regular, design: .rounded))
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .frame(width: 260, alignment: .leading)
        }
    }
}
