import AppKit
import SwiftUI

/// User-facing diagnostics window. Shows the current engine phase plus the
/// last few launch traces parsed from ``JunoLifecycleTraceLog``. Surfaced
/// from the menu bar ("Show Diagnostics…") and from the launch splash's
/// "View logs" button when the engine fails to start.
@MainActor
enum JunoDiagnosticsWindow {
    private static var windowController: NSWindowController?
    private static var closeDelegate: JunoActivationRestoringWindowDelegate?

    static func show() {
        if let existing = windowController, let w = existing.window {
            JunoWindowActivation.bringToFront(w)
            return
        }
        let view = JunoDiagnosticsView(lifecycle: JunoEngineLifecycle.shared)
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.title = "Juno · Diagnostics"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 760, height: 560))
        window.center()
        window.isReleasedWhenClosed = false
        let del = JunoActivationRestoringWindowDelegate {
            windowController = nil
            closeDelegate = nil
        }
        closeDelegate = del
        window.delegate = del
        let controller = NSWindowController(window: window)
        windowController = controller
        controller.showWindow(nil)
        JunoWindowActivation.bringToFront(window)
    }
}

// MARK: - View

struct JunoDiagnosticsView: View {
    @ObservedObject var lifecycle: JunoEngineLifecycle
    @State private var traces: [TraceFile] = []
    @State private var bundleURL: URL?
    @State private var bundleError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    currentPhaseCard
                    actionsRow
                    tracesSection
                }
                .padding(20)
            }
        }
        .background(Color(NSColor.windowBackgroundColor))
        .onAppear { reloadTraces() }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "stethoscope")
                .font(.title3)
            Text("Engine diagnostics")
                .font(.system(.title3, design: .rounded).weight(.semibold))
            Spacer()
            Button {
                reloadTraces()
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.borderless)
            .help("Reload traces")
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var currentPhaseCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Current phase")
                .font(.caption.weight(.semibold))
                .foregroundColor(.secondary)
            HStack(spacing: 10) {
                phaseDot
                Text(lifecycle.phase.label)
                    .font(.system(.title3, design: .monospaced))
                Spacer()
            }
            if !lifecycle.statusDetail.isEmpty {
                Text(lifecycle.statusDetail)
                    .font(.callout)
                    .foregroundColor(.secondary)
            }
            if let tail = lifecycle.lastCrashLogTail, !tail.isEmpty,
               case .failed = lifecycle.phase {
                CrashLogExpander(tail: tail)
                    .padding(.top, 4)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(NSColor.controlBackgroundColor))
        )
    }

    private var phaseDot: some View {
        Circle()
            .fill(phaseColor)
            .frame(width: 10, height: 10)
    }

    private var phaseColor: Color {
        switch lifecycle.phase {
        case .ready, .modelsLoaded: return .green
        case .needsModels, .needsPermissions: return .orange
        case .degraded: return .yellow
        case .failed: return .red
        case .healthOk, .socketBound: return JunoDesignTokens.accent
        default: return .gray
        }
    }

    private var actionsRow: some View {
        HStack(spacing: 10) {
            Button("Retry boot") { lifecycle.retry() }
                .disabled(!lifecycle.phase.isFailure)
            Button("Reset engine") { lifecycle.reset() }
            Button("Generate support bundle") { generateBundle() }
            Button("Open logs folder") { openLogsFolder() }
            Spacer()
            if bundleURL != nil {
                Text("Support bundle revealed in Finder")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .controlSize(.regular)
    }

    @ViewBuilder
    private var tracesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Launch traces")
                    .font(.headline)
                Spacer()
                if !traces.isEmpty {
                    Text("\(traces.count) file\(traces.count == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            if let err = bundleError {
                Text(err)
                    .font(.callout)
                    .foregroundColor(.red)
            }
            if traces.isEmpty {
                Text("No traces yet — restart Juno once and a trace will appear here.")
                    .font(.callout)
                    .foregroundColor(.secondary)
            } else {
                ForEach(traces) { trace in
                    TraceCard(trace: trace)
                }
            }
        }
    }

    // MARK: Actions

    private func reloadTraces() {
        let raw = JunoLifecycleTraceLog.shared.recentTraces()
        self.traces = raw.map { (url, events) in
            TraceFile(url: url, events: events)
        }
    }

    private func generateBundle() {
        if let url = JunoSupportBundle.generateAndReveal() {
            bundleURL = url
            bundleError = nil
        } else {
            bundleError = "Couldn't write support bundle to \(JunoSupportBundle.logDirectoryDisplayPath)."
        }
    }

    private func openLogsFolder() {
        JunoSupportBundle.revealLogDirectory()
    }
}

// MARK: - Crash log expander

private struct CrashLogExpander: View {
    let tail: String
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button(action: { expanded.toggle() }) {
                HStack(spacing: 6) {
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                        .font(.caption.weight(.semibold))
                    Text(expanded ? "Hide crash log" : "Show crash log")
                        .font(.caption.weight(.semibold))
                    Spacer()
                    Button("Copy") {
                        let pb = NSPasteboard.general
                        pb.clearContents()
                        pb.setString(tail, forType: .string)
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                }
            }
            .buttonStyle(.plain)
            .foregroundColor(.red)

            if expanded {
                ScrollView {
                    Text(tail)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(maxHeight: 220)
                .padding(8)
                .background(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(Color.red.opacity(0.08))
                )
            }
        }
    }
}

// MARK: - Trace card

private struct TraceFile: Identifiable {
    let url: URL
    let events: [JunoLifecycleTraceLog.Event]
    var id: URL { url }

    var firstEventDate: String {
        events.first?.ts ?? "unknown"
    }
    var totalDurationMs: Int {
        events.reduce(0) { $0 + $1.durMs }
    }
    var endedInFailure: Bool {
        events.last?.error != nil
    }
}

private struct TraceCard: View {
    let trace: TraceFile
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: { expanded.toggle() }) {
                HStack(spacing: 10) {
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                        .font(.caption.weight(.semibold))
                        .frame(width: 12)
                    statusDot
                    VStack(alignment: .leading, spacing: 2) {
                        Text(trace.firstEventDate)
                            .font(.system(.callout, design: .monospaced))
                        Text("\(trace.events.count) events · \(trace.totalDurationMs) ms total")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Button("Copy") { copyToClipboard() }
                        .buttonStyle(.borderless)
                        .controlSize(.small)
                }
            }
            .buttonStyle(.plain)
            if expanded {
                eventTable
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color(NSColor.controlBackgroundColor))
        )
    }

    private var statusDot: some View {
        Circle()
            .fill(trace.endedInFailure ? Color.red : Color.green)
            .frame(width: 8, height: 8)
    }

    private var eventTable: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(trace.events.enumerated()), id: \.offset) { _, event in
                HStack(alignment: .top, spacing: 8) {
                    Text(event.from)
                        .frame(width: 110, alignment: .trailing)
                    Text("→")
                    Text(event.to)
                        .frame(width: 130, alignment: .leading)
                    Text("\(event.durMs) ms")
                        .frame(width: 70, alignment: .trailing)
                        .foregroundColor(.secondary)
                    if let err = event.error {
                        Text(err)
                            .foregroundColor(.red)
                            .lineLimit(1)
                    } else if let note = event.note {
                        Text(note)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                    Spacer(minLength: 0)
                }
                .font(.system(.caption, design: .monospaced))
            }
        }
        .padding(.top, 8)
        .padding(.leading, 22)
    }

    private func copyToClipboard() {
        let lines = trace.events.map { ev in
            "\(ev.ts)  \(ev.from) → \(ev.to)  \(ev.durMs)ms  attempt=\(ev.attempt)" +
            (ev.error.map { "  ERROR: \($0)" } ?? "") +
            (ev.note.map { "  // \($0)" } ?? "")
        }
        let text = lines.joined(separator: "\n")
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
    }
}
