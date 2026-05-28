import Combine
import AppKit
import Foundation
import Sparkle

@MainActor
final class JunoUpdater: NSObject, ObservableObject, SPUUpdaterDelegate {
    static let shared = JunoUpdater()

    @Published private(set) var isConfigured = false
    @Published private(set) var isStarted = false
    @Published private(set) var canCheckForUpdates = false
    @Published private(set) var updateAvailable = false
    @Published private(set) var latestVersion: String?
    @Published private(set) var lastStatus = "Updates are not configured for this build."
    @Published private(set) var lastError: String?
    @Published private(set) var automaticallyChecksForUpdates = false
    @Published private(set) var automaticallyDownloadsUpdates = false
    @Published private(set) var allowsAutomaticUpdates = false
    @Published private(set) var lastUpdateCheckDate: Date?

    private var updaterController: SPUStandardUpdaterController?
    private var configuredFeedURL: URL?
    private var configuredChannel: String?

    private override init() {
        super.init()
        reloadConfiguration()
    }

    var statusLine: String {
        if !isConfigured { return lastStatus }
        if let latestVersion, updateAvailable {
            return "Version \(latestVersion) is ready to install."
        }
        if let lastUpdateCheckDate {
            let formatter = RelativeDateTimeFormatter()
            formatter.unitsStyle = .full
            return "Last checked \(formatter.localizedString(for: lastUpdateCheckDate, relativeTo: Date()))."
        }
        return "Automatic update checks are ready."
    }

    func startIfConfigured() {
        guard updaterController == nil else {
            syncStateFromUpdater()
            return
        }
        reloadConfiguration()
        guard isConfigured else { return }

        let controller = SPUStandardUpdaterController(
            startingUpdater: false,
            updaterDelegate: self,
            userDriverDelegate: nil
        )
        updaterController = controller

        do {
            try controller.updater.start()
            isStarted = true
            lastError = nil
            lastStatus = "Automatic update checks are ready."
        } catch {
            isStarted = false
            lastError = error.localizedDescription
            lastStatus = "Updater could not start."
            NSLog("Juno: Sparkle updater failed to start: \(error.localizedDescription)")
        }
        syncStateFromUpdater()
    }

    func checkForUpdates() {
        startIfConfigured()
        guard let updaterController, isStarted else {
            showConfigurationAlertIfNeeded()
            return
        }
        syncStateFromUpdater()
        guard canCheckForUpdates else { return }
        updaterController.checkForUpdates(nil)
    }

    func refreshState() {
        syncStateFromUpdater()
    }

    func setAutomaticallyChecksForUpdates(_ enabled: Bool) {
        startIfConfigured()
        guard let updater = updaterController?.updater else { return }
        updater.automaticallyChecksForUpdates = enabled
        syncStateFromUpdater()
    }

    func setAutomaticallyDownloadsUpdates(_ enabled: Bool) {
        startIfConfigured()
        guard let updater = updaterController?.updater, updater.allowsAutomaticUpdates else { return }
        updater.automaticallyDownloadsUpdates = enabled
        syncStateFromUpdater()
    }

    func feedURLString(for updater: SPUUpdater) -> String? {
        configuredFeedURL?.absoluteString
    }

    func allowedChannels(for updater: SPUUpdater) -> Set<String> {
        guard let channel = configuredChannel, channel != "stable" else { return [] }
        return [channel]
    }

    func updater(_ updater: SPUUpdater, didFindValidUpdate item: SUAppcastItem) {
        updateAvailable = true
        latestVersion = item.displayVersionString
        lastError = nil
        lastStatus = "Update available."
        syncStateFromUpdater()
    }

    func updaterDidNotFindUpdate(_ updater: SPUUpdater, error: any Error) {
        updateAvailable = false
        latestVersion = nil
        lastError = nil
        lastStatus = "Juno is up to date."
        syncStateFromUpdater()
    }

    func updater(_ updater: SPUUpdater, didAbortWithError error: any Error) {
        lastError = error.localizedDescription
        lastStatus = "Update check failed."
        syncStateFromUpdater()
    }

    func updater(_ updater: SPUUpdater, didFinishUpdateCycleFor updateCheck: SPUUpdateCheck, error: (any Error)?) {
        if let error {
            lastError = error.localizedDescription
            lastStatus = "Update check failed."
        } else if updateAvailable {
            lastStatus = "Update available."
        }
        syncStateFromUpdater()
    }

    private func reloadConfiguration() {
        let info = Bundle.main.infoDictionary ?? [:]
        let explicitEnabled = info["JunoOTAEnabled"] as? Bool
        let rawFeed = (info["SUFeedURL"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let rawKey = (info["SUPublicEDKey"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let channel = (info["JunoUpdateChannel"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)

        configuredFeedURL = URL(string: rawFeed)
        configuredChannel = channel?.isEmpty == false ? channel : nil
        isConfigured = (explicitEnabled ?? true)
            && configuredFeedURL != nil
            && !rawKey.isEmpty
            && !rawFeed.contains("example.invalid")

        if isConfigured {
            lastStatus = "Automatic update checks are ready."
        } else {
            lastStatus = "Updates are not configured for this build."
        }
    }

    private func syncStateFromUpdater() {
        guard let updater = updaterController?.updater else {
            canCheckForUpdates = false
            automaticallyChecksForUpdates = false
            automaticallyDownloadsUpdates = false
            allowsAutomaticUpdates = false
            lastUpdateCheckDate = nil
            return
        }
        canCheckForUpdates = updater.canCheckForUpdates
        automaticallyChecksForUpdates = updater.automaticallyChecksForUpdates
        automaticallyDownloadsUpdates = updater.automaticallyDownloadsUpdates
        allowsAutomaticUpdates = updater.allowsAutomaticUpdates
        lastUpdateCheckDate = updater.lastUpdateCheckDate
    }

    private func showConfigurationAlertIfNeeded() {
        guard !isConfigured else { return }
        let alert = NSAlert()
        alert.messageText = "Updates are not configured in this build"
        alert.informativeText = "Build Juno with an appcast URL and Sparkle public EdDSA key to enable over-the-air updates."
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}
