import AppKit
import SwiftUI

// MARK: - Shared card (onboarding + settings)

/// Explains the local voice engine in plain language. Reachable from the
/// menu bar; must remain free of Python / shell / "workbench" terminology.
/// All visible strings come from `JunoEngineHelpCopy` so naming-discipline
/// regressions are caught by JunoBrokerHelpCopyTests.
struct JunoBrokerSetupCard: View {
    @State private var brokerReachable = false
    /// Retained as a parameter for source compatibility with prior callers.
    /// No longer toggles a developer disclosure (the developer affordances
    /// were removed in fix #6).
    var showDoctorHint: Bool = true
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: brokerReachable ? "checkmark.circle.fill" : "waveform.circle")
                        .font(.title2)
                        .foregroundStyle(brokerReachable ? .green : .orange)
                    VStack(alignment: .leading, spacing: 6) {
                        Text(brokerReachable
                             ? JunoEngineHelpCopy.headlineConnected
                             : JunoEngineHelpCopy.headlineDisconnected)
                            .font(.headline)
                        if brokerReachable {
                            Text(JunoEngineHelpCopy.connectedExplanation)
                                .font(.callout)
                                .foregroundStyle(JunoTheme.secondaryText(scheme))
                                .fixedSize(horizontal: false, vertical: true)
                        } else {
                            Text(JunoEngineHelpCopy.disconnectedExplanation)
                                .font(.callout)
                                .foregroundStyle(JunoTheme.secondaryText(scheme))
                                .fixedSize(horizontal: false, vertical: true)
                            Button(JunoEngineHelpCopy.checkAgainButton) { refreshHealth() }
                                .junoPrimaryActionButton()
                        }
                    }
                }

                if !brokerReachable {
                    VStack(alignment: .leading, spacing: 8) {
                        Label {
                            Text(JunoEngineHelpCopy.tryThisFirstHeading)
                                .font(.subheadline.weight(.semibold))
                        } icon: {
                            Image(systemName: "1.circle")
                        }
                        Text(JunoEngineHelpCopy.tryThisFirstBody)
                            .font(.callout)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .fixedSize(horizontal: false, vertical: true)

                        Label {
                            Text(JunoEngineHelpCopy.stillStuckHeading)
                                .font(.subheadline.weight(.semibold))
                        } icon: {
                            Image(systemName: "2.circle")
                        }
                        Text(JunoEngineHelpCopy.stillStuckBody)
                            .font(.callout)
                            .foregroundStyle(JunoTheme.secondaryText(scheme))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.top, 4)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            HStack(spacing: 8) {
                JunoCommaMark(color: Color.accentColor, scale: 0.34)
                    .frame(width: 22, height: 28)
                Text("Voice engine")
                    .font(.headline)
            }
        }
        .onAppear { refreshHealth() }
    }

    private func refreshHealth() {
        JunoBroker.pingHealth { ok in
            DispatchQueue.main.async {
                brokerReachable = ok
            }
        }
    }
}

// MARK: - Floating help window (menu bar)

private struct JunoBrokerHelpRootView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            JunoChromeBrandHeader(
                style: .compact,
                title: "Voice engine",
                subtitle: JunoEngineHelpCopy.helpWindowSubtitle
            )
            JunoBrokerSetupCard(showDoctorHint: true)
        }
        .padding(20)
        .frame(minWidth: 420)
        .junoBrandWindow()
    }
}

enum JunoBrokerHelpWindow {
    private static var windowController: NSWindowController?

    @MainActor
    static func show() {
        if let wc = windowController, let w = wc.window, w.isVisible {
            JunoWindowActivation.bringToFront(w)
            return
        }
        windowController = nil
        let hosting = NSHostingController(rootView: JunoBrokerHelpRootView())
        let window = NSWindow(contentViewController: hosting)
        window.title = JunoEngineHelpCopy.helpWindowTitle
        window.styleMask = [.titled, .closable]
        window.standardWindowButton(.miniaturizeButton)?.isHidden = true
        window.standardWindowButton(.zoomButton)?.isHidden = true
        window.center()
        window.isReleasedWhenClosed = false
        let wc = NSWindowController(window: window)
        windowController = wc
        wc.showWindow(nil)
        JunoWindowActivation.bringToFront(window)
    }
}
