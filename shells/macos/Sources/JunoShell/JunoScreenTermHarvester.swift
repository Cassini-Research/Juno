import AppKit
import CoreGraphics
import ScreenCaptureKit
import Vision

/// On-screen vocabulary, app-agnostic.
///
/// AX-tree text extraction returns nothing for large classes of apps
/// (Electron, web views, canvases, terminals with custom renderers), which
/// left `candidate_entities` empty in production and made screen-grounded
/// ASR correction a fiction. This harvester screenshots the displays and
/// OCRs them (Vision, fast pass, language correction OFF so identifiers
/// like "Qwen" are not "corrected" away), then distills the distinctive
/// terms — names, identifiers, product words — that dictation is likely to
/// mishear. It runs only while a dictation session is active, visible-screen
/// context is enabled, and macOS access is already granted. It never requests
/// access from the dictation path.
final class JunoScreenTermHarvester {
    static let shared = JunoScreenTermHarvester()

    private let queue = DispatchQueue(label: "juno.screen.terms", qos: .utility)
    private let lock = NSLock()
    private var timer: DispatchSourceTimer?
    private var active = false
    private var lastTerms: [String] = []
    private var lastRefreshAt: Date = .distantPast

    // MARK: lifecycle

    func activate() {
        guard JunoScreenContextAccess.isEnabledAndGranted else {
            clearTerms()
            return
        }
        lock.lock()
        let alreadyActive = active
        active = true
        lock.unlock()
        if alreadyActive {
            queue.async { [weak self] in self?.refresh() }
            return
        }
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: 6.0, leeway: .seconds(1))
        t.setEventHandler { [weak self] in self?.refresh() }
        t.resume()
        lock.lock()
        timer?.cancel()
        timer = t
        lock.unlock()
    }

    func deactivate() {
        lock.lock()
        active = false
        timer?.cancel()
        timer = nil
        lock.unlock()
    }

    /// Last harvested terms — non-blocking; safe from any thread.
    func currentTerms() -> [String] {
        guard JunoScreenContextAccess.isEnabledAndGranted else { return [] }
        lock.lock()
        defer { lock.unlock() }
        // Terms older than a minute are stale context from a previous turn.
        if Date().timeIntervalSince(lastRefreshAt) > 60 { return [] }
        return lastTerms
    }

    // MARK: capture + OCR

    private func refresh() {
        guard JunoScreenContextAccess.isEnabledAndGranted else {
            clearTerms()
            return
        }
        let images = Self.captureDisplays()
        guard !images.isEmpty else { return }
        var lines: [String] = []
        for image in images {
            lines.append(contentsOf: Self.recognizeText(in: image))
        }
        let terms = Self.distillTerms(from: lines)
        lock.lock()
        lastTerms = terms
        lastRefreshAt = Date()
        lock.unlock()
    }

    private func clearTerms() {
        lock.lock()
        lastTerms = []
        lastRefreshAt = .distantPast
        lock.unlock()
    }

    private static func captureDisplays() -> [CGImage] {
        let semaphore = DispatchSemaphore(value: 0)
        var content: SCShareableContent?
        SCShareableContent.getExcludingDesktopWindows(true, onScreenWindowsOnly: true) { c, _ in
            content = c
            semaphore.signal()
        }
        _ = semaphore.wait(timeout: .now() + 3)
        guard let shareable = content else { return [] }
        let ownBundle = Bundle.main.bundleIdentifier ?? ""
        let excluded = shareable.applications.filter { $0.bundleIdentifier == ownBundle }
        var images: [CGImage] = []
        for display in shareable.displays.prefix(2) {
            let filter = SCContentFilter(
                display: display,
                excludingApplications: excluded,
                exceptingWindows: []
            )
            let config = SCStreamConfiguration()
            // Half-resolution OCRs UI text reliably on Retina and is ~4×
            // faster than full pixel size.
            config.width = display.width
            config.height = display.height
            config.showsCursor = false
            let imgSemaphore = DispatchSemaphore(value: 0)
            var captured: CGImage?
            SCScreenshotManager.captureImage(contentFilter: filter, configuration: config) { image, _ in
                captured = image
                imgSemaphore.signal()
            }
            _ = imgSemaphore.wait(timeout: .now() + 3)
            if let captured { images.append(captured) }
        }
        return images
    }

    private static func recognizeText(in image: CGImage) -> [String] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .fast
        // Critical: no language correction — it would "fix" the exact
        // identifiers (Qwen, NabloGrid, juno_v2) we exist to preserve.
        request.usesLanguageCorrection = false
        request.recognitionLanguages = ["en-US"]
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        guard (try? handler.perform([request])) != nil else { return [] }
        let observations = request.results ?? []
        return observations.compactMap { $0.topCandidates(1).first?.string }
    }

    // MARK: distillation

    private static let uiChrome: Set<String> = [
        "file", "edit", "view", "window", "help", "close", "save", "cancel",
        "open", "search", "settings", "menu", "back", "next", "done", "new",
        "today", "inbox", "untitled", "loading", "submit", "delete", "share",
        "copy", "paste", "undo", "redo", "cut", "print", "quit", "about",
        "preferences", "tools", "format", "insert", "table", "home", "end",
        "page", "tab", "enter", "return", "shift", "control", "option",
        "command", "escape", "online", "offline", "notifications", "profile",
        "account", "login", "logout", "signin", "signout", "yes", "no",
    ]

    private static let commonWords: Set<String> = [
        "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
        "had", "has", "have", "her", "him", "his", "how", "its", "may", "our",
        "out", "she", "they", "this", "that", "was", "were", "will", "with",
        "what", "when", "where", "which", "while", "who", "why", "your",
        "from", "into", "over", "under", "about", "after", "before", "between",
        "because", "been", "being", "both", "each", "few", "more", "most",
        "other", "some", "such", "than", "then", "there", "these", "those",
        "through", "very", "just", "also", "only", "even", "still", "should",
        "would", "could", "make", "made", "like", "time", "than", "them",
        "now", "get", "got", "one", "two", "use", "used", "using", "way",
        "see", "say", "said", "did", "does", "doing", "done", "here", "their",
        "look", "want", "give", "first", "last", "long", "good", "great",
        "little", "own", "same", "right", "left", "old", "tell", "work",
        "things", "thing", "day", "days", "back", "off", "let", "too", "many",
        "much", "every", "again", "down", "need", "well", "people", "text",
        "note", "notes", "words", "word",
    ]

    static func distillTerms(from lines: [String]) -> [String] {
        var scores: [String: (score: Int, display: String)] = [:]

        func add(_ term: String, weight: Int) {
            let trimmed = term.trimmingCharacters(in: CharacterSet(charactersIn: " .,;:!?()[]{}<>\"'`"))
            guard trimmed.count >= 2, trimmed.count <= 40 else { return }
            let key = trimmed.lowercased()
            guard !uiChrome.contains(key), !commonWords.contains(key) else { return }
            // Pure numbers carry no recognition value.
            guard trimmed.rangeOfCharacter(from: .letters) != nil else { return }
            let existing = scores[key]
            scores[key] = ((existing?.score ?? 0) + weight, existing?.display ?? trimmed)
        }

        let tokenPattern = try? NSRegularExpression(pattern: "[A-Za-z][A-Za-z0-9_'./-]{1,39}")
        for line in lines {
            guard let tokenPattern else { break }
            let ns = line as NSString
            let matches = tokenPattern.matches(in: line, range: NSRange(location: 0, length: ns.length))
            var previousCapWord: String?
            for match in matches {
                let token = ns.substring(with: match.range)
                let lower = token.lowercased()
                let hasDigit = token.rangeOfCharacter(from: .decimalDigits) != nil
                let hasUnderscoreOrPath = token.contains("_") || token.contains("/") || token.contains(".")
                let isAllCaps = token.count >= 2 && token.count <= 10 && token == token.uppercased() && !hasDigit
                let body = String(token.dropFirst())
                let isCamel = body.rangeOfCharacter(from: .uppercaseLetters) != nil && body.rangeOfCharacter(from: .lowercaseLetters) != nil
                let isCapitalized = token.first.map { $0.isUppercase } == true && token.dropFirst().allSatisfy { $0.isLowercase }

                if hasDigit && token.rangeOfCharacter(from: .letters) != nil {
                    add(token, weight: 3)
                } else if isCamel || hasUnderscoreOrPath {
                    add(token, weight: 3)
                } else if isAllCaps {
                    add(token, weight: 2)
                } else if isCapitalized && !commonWords.contains(lower) && token.count >= 3 {
                    add(token, weight: 1)
                    if let prev = previousCapWord {
                        add("\(prev) \(token)", weight: 2)
                    }
                }
                previousCapWord = isCapitalized ? token : nil
            }
        }

        return scores.values
            .sorted { $0.score > $1.score || ($0.score == $1.score && $0.display < $1.display) }
            .prefix(40)
            .map { $0.display }
    }
}
