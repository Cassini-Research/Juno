// JunoUninstaller.swift
//
// Complete, in-app uninstall: stops the engine, removes every artifact
// Juno leaves on disk (Application Support, logs, caches, downloaded
// models, LaunchAgents, login item), moves the app to the Trash, and
// hands the work that can't happen while the app is alive (UserDefaults,
// TCC grants, saved window state) to a detached post-quit script.
//
// Deleting preferences in-process is a trap: cfprefsd keeps the domain
// in memory and re-persists it when the app exits, which silently
// resurrects `onboardingCompleted` — the same class of "uninstalled but
// it still remembers me" bug this feature exists to fix. Everything in
// the preferences/TCC bucket therefore runs *after* our PID exits.

import AppKit
import Foundation
import ServiceManagement
import SwiftUI

enum JunoUninstaller {

    /// Models the default engine profile downloads into the shared HF
    /// cache. Used as the floor for deletion; live repo ids reported by
    /// the broker are merged on top so profile changes are covered.
    /// Other tools' models in the same cache are never touched.
    static let knownModelRepoIds: [String] = [
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "mlx-community/Qwen3-0.6B-4bit",
        "mlx-community/parakeet-tdt-0.6b-v3",
    ]

    static let launchAgentLabels: [String] = [
        "com.juno.voice-engine",
        "com.juno.launch",        // legacy
        "com.juno.shell.agent",   // legacy
    ]

    struct Plan: Equatable {
        var dataPaths: [URL] = []
        var modelPaths: [URL] = []
        var launchAgentPlists: [URL] = []
        /// Set only when the running bundle is removable (installed in
        /// /Applications). Dev builds run from a repo checkout and must
        /// never delete themselves.
        var appBundle: URL?
    }

    static func hfModelCacheDir(
        repoId: String,
        hfHome: String?,
        home: URL
    ) -> URL {
        let hubRoot: URL
        if let hfHome, !hfHome.isEmpty {
            hubRoot = URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        } else {
            hubRoot = home.appendingPathComponent(".cache/huggingface/hub")
        }
        let safe = "models--" + repoId.replacingOccurrences(of: "/", with: "--")
        return hubRoot.appendingPathComponent(safe)
    }

    static func plan(
        extraRepoIds: [String] = [],
        home: URL = FileManager.default.homeDirectoryForCurrentUser,
        bundleURL: URL = Bundle.main.bundleURL,
        hfHome: String? = ProcessInfo.processInfo.environment["HF_HOME"]
    ) -> Plan {
        var plan = Plan()
        let lib = home.appendingPathComponent("Library")

        plan.dataPaths = [
            lib.appendingPathComponent("Application Support/com.juno.shell"),
            lib.appendingPathComponent("Application Support/Juno"), // pre-bundle-id legacy
            lib.appendingPathComponent("Logs/Juno"),
            lib.appendingPathComponent("Caches/com.juno.shell"),
            lib.appendingPathComponent("HTTPStorages/com.juno.shell"),
            lib.appendingPathComponent("WebKit/com.juno.shell"),
            lib.appendingPathComponent("Saved Application State/com.juno.shell.savedState"),
        ]

        var repoIds = knownModelRepoIds
        for repo in extraRepoIds {
            let trimmed = repo.trimmingCharacters(in: .whitespaces)
            if trimmed.contains("/"), !repoIds.contains(trimmed) {
                repoIds.append(trimmed)
            }
        }
        plan.modelPaths = repoIds.map {
            hfModelCacheDir(repoId: $0, hfHome: hfHome, home: home)
        }

        plan.launchAgentPlists = launchAgentLabels.map {
            lib.appendingPathComponent("LaunchAgents/\($0).plist")
        }

        // Self-removal only for a real install. Anything else (repo
        // checkout, dist/, a mounted DMG) stays put.
        if bundleURL.path.hasPrefix("/Applications/") {
            plan.appBundle = bundleURL
        }
        return plan
    }

    /// Stop the launchd-managed engine and any engine processes spawned
    /// directly from this bundle. Must run before file deletion so the
    /// engine doesn't recreate runtime state mid-wipe.
    private static func stopEngine() {
        for label in launchAgentLabels {
            runQuietly("/bin/launchctl", ["bootout", "gui/\(getuid())/\(label)"])
        }
        // Engine processes started by the app itself (bootstrap path)
        // rather than launchd. Match on our bundle's engine directory so
        // we never touch unrelated python processes.
        runQuietly("/usr/bin/pkill", ["-f", "Juno.app/Contents/Resources/engine"])
    }

    @discardableResult
    private static func runQuietly(_ launchPath: String, _ arguments: [String]) -> Bool {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launchPath)
        p.arguments = arguments
        p.standardOutput = Pipe()
        p.standardError = Pipe()
        do {
            try p.run()
            p.waitUntilExit()
            return p.terminationStatus == 0
        } catch {
            return false
        }
    }

    /// Work that must happen after our PID exits: preferences (cfprefsd
    /// re-persists the domain at app exit), TCC grants, saved window
    /// state, and any plist a teardown race recreated. Spawned detached
    /// so it survives NSApp.terminate.
    private static func spawnPostQuitCleanup(plan: Plan) {
        let supportDir = NSHomeDirectory() + "/Library/Application Support/com.juno.shell"
        let savedState = NSHomeDirectory() + "/Library/Saved Application State/com.juno.shell.savedState"
        let logsDir = NSHomeDirectory() + "/Library/Logs/Juno"
        let cachesDir = NSHomeDirectory() + "/Library/Caches/com.juno.shell"
        let agentPlists = plan.launchAgentPlists.map { "rm -f \"\($0.path)\"" }.joined(separator: "\n")
        // Logs/support are re-swept here because app teardown can recreate
        // them *after* the in-app wipe: killing the engine makes the
        // supervisor write a crash snapshot into Logs/Juno (observed in
        // live testing), and quit-path logging can do the same.
        let script = """
        #!/bin/bash
        # Juno post-quit uninstall cleanup. Self-deletes when done.
        for _ in $(seq 1 60); do
          kill -0 \(ProcessInfo.processInfo.processIdentifier) 2>/dev/null || break
          sleep 0.5
        done
        defaults delete com.juno.shell >/dev/null 2>&1
        rm -f "$HOME/Library/Preferences/com.juno.shell.plist"
        tccutil reset All com.juno.shell >/dev/null 2>&1
        \(agentPlists)
        rm -rf "\(savedState)"
        rm -rf "\(supportDir)"
        rm -rf "\(logsDir)"
        rm -rf "\(cachesDir)"
        rm -f "$0"
        """
        let scriptURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("juno-uninstall-\(UUID().uuidString).sh")
        do {
            try script.write(to: scriptURL, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o700], ofItemAtPath: scriptURL.path
            )
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/bin/bash")
            p.arguments = [scriptURL.path]
            try p.run()  // intentionally not waited on
        } catch {
            NSLog("Juno: failed to spawn post-quit cleanup: \(error.localizedDescription)")
        }
    }

    /// Returns a short human-readable summary of anything that could not
    /// be removed (nil when fully clean). Runs synchronously off-main.
    static func performUninstall(extraRepoIds: [String]) -> String? {
        let fm = FileManager.default
        let plan = plan(extraRepoIds: extraRepoIds)
        var failures: [String] = []

        stopEngine()

        for url in plan.dataPaths + plan.modelPaths {
            guard fm.fileExists(atPath: url.path) else { continue }
            do {
                try fm.removeItem(at: url)
            } catch {
                failures.append(url.lastPathComponent)
            }
        }

        if #available(macOS 13.0, *) {
            try? SMAppService.mainApp.unregister()
        }

        if let appBundle = plan.appBundle {
            do {
                try fm.trashItem(at: appBundle, resultingItemURL: nil)
            } catch {
                failures.append("Juno.app (move it to the Trash manually)")
            }
        }

        spawnPostQuitCleanup(plan: plan)

        return failures.isEmpty ? nil : "Could not remove: " + failures.joined(separator: ", ")
    }
}

// MARK: - Settings card

/// Always-visible uninstall affordance at the bottom of Settings.
/// Deliberately NOT behind Developer mode: the people who most need a
/// clean uninstall are end users, not developers.
struct JunoUninstallSettingsCard: View {
    @Environment(\.colorScheme) private var scheme
    @State private var pendingUninstall = false
    @State private var isUninstalling = false
    @State private var uninstallStatus: String?

    var body: some View {
        JunoPreferenceSection(
            title: "Uninstall Juno",
            subtitle: "Removes everything Juno put on this Mac: downloaded voice models (~4 GB), dictation history and recordings, learned memory, settings, the background voice engine, and the app itself. Nothing is kept."
        ) {
            HStack(spacing: 10) {
                Button(isUninstalling ? "Uninstalling…" : "Uninstall Juno…", role: .destructive) {
                    pendingUninstall = true
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .tint(JunoDesignTokens.danger)
                .disabled(isUninstalling)

                if isUninstalling {
                    ProgressView()
                        .controlSize(.small)
                        .scaleEffect(0.75)
                }

                Spacer(minLength: 0)
            }

            if let uninstallStatus {
                Text(uninstallStatus)
                    .font(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .confirmationDialog(
            "Uninstall Juno from this Mac?",
            isPresented: $pendingUninstall,
            titleVisibility: .visible
        ) {
            Button("Uninstall and Quit", role: .destructive) { runUninstall() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                "This deletes the downloaded voice models, your dictation history and recordings, learned memory, and all settings, then moves Juno to the Trash and quits. This cannot be undone."
            )
        }
    }

    private func runUninstall() {
        guard !isUninstalling else { return }
        isUninstalling = true
        uninstallStatus = "Stopping the voice engine and removing data…"

        // Best-effort: ask the (still running) broker which model repos
        // the active profile uses, so non-default profiles get their
        // models removed too. A dead broker just means we fall back to
        // the known default repos inside JunoUninstaller.
        JunoBroker.fetchSetupStatus { result in
            var extraRepoIds: [String] = []
            if case .success(let s) = result {
                extraRepoIds = [s.previewRepoId, s.finalRepoId, s.writerRepoId, s.liveCorrectorRepoId]
                    .compactMap { $0 }
                    .filter { !$0.isEmpty }
            }
            DispatchQueue.global(qos: .userInitiated).async {
                let failureSummary = JunoUninstaller.performUninstall(extraRepoIds: extraRepoIds)
                DispatchQueue.main.async {
                    isUninstalling = false
                    let alert = NSAlert()
                    alert.alertStyle = .informational
                    if let failureSummary {
                        alert.messageText = "Juno is uninstalled, with one exception"
                        alert.informativeText = failureSummary + "\n\nJuno will quit now."
                    } else {
                        alert.messageText = "Juno has been uninstalled"
                        alert.informativeText = "All models, history, and settings were removed. Juno will quit now."
                    }
                    alert.addButton(withTitle: "Quit")
                    alert.runModal()
                    NSApp.terminate(nil)
                }
            }
        }
    }
}
