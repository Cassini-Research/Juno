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
        "accessibility", "onboarding", "permission", "permissions", "reminder",
        "reminders",
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
            guard !isLikelyOCRJunk(trimmed) else { return }
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
                let isCamel = Self.isTwoPartCamelOrAcronymToken(token)
                let isCapitalized = Self.isSimpleCapitalizedToken(token)

                if hasDigit && token.rangeOfCharacter(from: .letters) != nil && !hasUnderscoreOrPath {
                    previousCapWord = isCapitalized ? token : nil
                    continue
                } else if isCamel || (hasUnderscoreOrPath && Self.isTechnicalScreenIdentifier(token)) {
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

    private static func isLikelyOCRJunk(_ token: String) -> Bool {
        let hasLetter = token.rangeOfCharacter(from: .letters) != nil
        let hasDigit = token.rangeOfCharacter(from: .decimalDigits) != nil
        let hasIdentifierSeparator = token.rangeOfCharacter(from: CharacterSet(charactersIn: "_./-#")) != nil
        if hasLetter && hasDigit && !hasIdentifierSeparator {
            return true
        }
        let letters = token.filter { $0.isLetter }
        if token.count <= 4,
           !letters.isEmpty,
           token != token.uppercased(),
           token != token.lowercased(),
           token != token.capitalized {
            return true
        }
        let normalized = ocrNoiseKey(token)
        if uiChrome.contains(normalized) || commonWords.contains(normalized) {
            return true
        }
        if isNearOCRNoiseWord(normalized) {
            return true
        }
        return false
    }

    private static func ocrNoiseKey(_ token: String) -> String {
        let map: [Character: Character] = [
            "0": "o",
            "1": "l",
            "3": "e",
            "4": "a",
            "5": "s",
            "7": "t",
            "8": "b",
            "9": "g",
        ]
        return String(token.lowercased().map { map[$0] ?? $0 })
    }

    private static func isNearOCRNoiseWord(_ normalized: String) -> Bool {
        guard normalized.count >= 6 else { return false }
        for word in uiChrome.union(commonWords) {
            guard word.count >= 6 else { continue }
            guard abs(normalized.count - word.count) <= 2 else { continue }
            if editDistance(normalized, word, maxDistance: 2) <= 2 {
                return true
            }
        }
        return false
    }

    private static func editDistance(_ lhs: String, _ rhs: String, maxDistance: Int) -> Int {
        let a = Array(lhs)
        let b = Array(rhs)
        if abs(a.count - b.count) > maxDistance { return maxDistance + 1 }
        var previous = Array(0...b.count)
        for i in 1...a.count {
            var current = [i]
            var rowMin = i
            for j in 1...b.count {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                let value = min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost
                )
                current.append(value)
                rowMin = min(rowMin, value)
            }
            if rowMin > maxDistance { return maxDistance + 1 }
            previous = current
        }
        return previous.last ?? maxDistance + 1
    }

    private static func isSimpleCapitalizedToken(_ token: String) -> Bool {
        let letters = token.filter { $0.isLetter }
        guard letters.count >= 3, let first = letters.first, first.isUppercase else { return false }
        return letters.dropFirst().allSatisfy { $0.isLowercase }
    }

    private static func isTwoPartCamelOrAcronymToken(_ token: String) -> Bool {
        let letters = String(token.filter { $0.isLetter })
        let patterns = [
            "^[A-Z][a-z]{2,}[A-Z][a-z]{2,}$",
            "^[A-Z][a-z]{2,}[A-Z]{2,4}$",
        ]
        return patterns.contains { pattern in
            guard let regex = try? NSRegularExpression(pattern: pattern) else { return false }
            let range = NSRange(letters.startIndex..., in: letters)
            return regex.firstMatch(in: letters, range: range) != nil
        }
    }

    private static func isTechnicalScreenIdentifier(_ token: String) -> Bool {
        let clean = token.trimmingCharacters(in: CharacterSet(charactersIn: " .,;:!?()[]{}<>\"'`"))
        let patterns = [
            "^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)+(?:\\.[a-z0-9]{1,8})?$",
            "^[a-z][a-z0-9]*(?:\\.[a-z0-9]{1,8})$",
            "^[A-Z]{2,}\\d{1,6}$",
        ]
        return patterns.contains { pattern in
            guard let regex = try? NSRegularExpression(pattern: pattern) else { return false }
            let range = NSRange(clean.startIndex..., in: clean)
            return regex.firstMatch(in: clean, range: range) != nil
        }
    }
}
