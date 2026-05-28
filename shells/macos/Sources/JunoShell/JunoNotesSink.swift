// JunoNotesSink.swift
//
// AppleScript bridge to the macOS Notes app. Used by ``JunoActionExecutor``
// for ``ActionKind.note`` requests.
//
// **Destination**: notes are filed in a folder called **Juno**, under the
// iCloud account when present (else the first available account, then
// "On My Mac"). The HUD/History UI tells the user this explicitly so
// they don't dictate a note, see "Saved", and then go hunting.
//
// **Why AppleScript and not a private framework?** Apple ships no public
// EventKit-style API for Notes. AppleScript via NSAppleScript / osascript
// is the only sanctioned path; it requires Automation permission for the
// Notes app (NSAppleEventsUsageDescription is set in JunoShellInfo.plist).
//
// **What we save**: a single note in the **Juno** folder (created on demand)
// with the body the parser extracted plus an optional "Captured with Juno"
// signature line. The signature is gated on
// ``JunoUserDefaults.actionsNotesSignatureEnabled`` (default ON) so users who
// dislike footers can turn it off in Settings → Voice Actions.
//
// **Hard rule:** never block the caller. Every public method takes a
// completion handler and runs the AppleScript on a background queue. Errors
// are surfaced as ``Result.failure`` with a human-readable message so the
// HUD chip can show "couldn't save".

import Foundation
import os.log

private let notesSinkLog = OSLog(subsystem: "com.juno.shell", category: "actions.notes")

/// Display name for the Notes folder Juno writes to. Centralized so HUD
/// strings, History strings, and the AppleScript stay in sync.
let JunoNotesFolderName = "Juno"

/// Stable fallback URL that just opens Notes.app. Used when AppleScript
/// returned no usable identifier — the per-note deep link can't be
/// formed, but we can still hand the user a one-click "Open Notes"
/// affordance instead of nothing.
let JunoNotesAppFallbackURL = URL(string: "notes://")!

final class JunoNotesSink {

    static let shared = JunoNotesSink()

    struct CreatedNote {
        /// Note id reported back by Notes.app (a long URL-shaped string).
        let id: String
        /// Deep link the HUD chip clicks to open the note. Notes uses
        /// ``applenotes://showNote?identifier=…`` for in-app deep links;
        /// when the id is unavailable we fall back to the app URL.
        let url: URL?
    }

    enum SinkError: Error, LocalizedError {
        case scriptError(String)
        case notInstalled
        case automationDenied

        var errorDescription: String? {
            switch self {
            case .scriptError(let msg): return msg
            case .notInstalled: return "Notes.app not installed."
            case .automationDenied:
                return "Juno needs permission to control Notes. Open System Settings → Privacy & Security → Automation and enable Notes for Juno."
            }
        }
    }

    private let queue = DispatchQueue(label: "juno.notes-sink", qos: .userInitiated)

    /// Create a new note in the **Juno** folder of the **iCloud** account
    /// (or the default account if iCloud is not configured). The signature
    /// flag toggles a trailing "— Captured with Juno · {time}" line.
    func createNote(
        body: String,
        appendSignature: Bool,
        folderName: String? = nil,
        completion: @escaping (Result<CreatedNote, SinkError>) -> Void
    ) {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            DispatchQueue.main.async {
                completion(.failure(.scriptError("note body was empty")))
            }
            return
        }
        let finalBody = appendSignature ? Self.applySignature(to: trimmed) : trimmed

        queue.async {
            let outcome = Self.runCreateScript(body: finalBody, folderName: folderName)
            DispatchQueue.main.async { completion(outcome) }
        }
    }

    func deleteNote(
        id: String,
        completion: @escaping (Result<Void, SinkError>) -> Void
    ) {
        let trimmed = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            DispatchQueue.main.async {
                completion(.failure(.scriptError("note id was empty")))
            }
            return
        }
        queue.async {
            let outcome = Self.runDeleteScript(id: trimmed)
            DispatchQueue.main.async { completion(outcome) }
        }
    }

    // MARK: - AppleScript

    /// AppleScript to create a note. The script:
    /// 1. Activates Notes.app silently (no window pop).
    /// 2. Picks a default account (first iCloud, falling back to "On My Mac").
    /// 3. Ensures a "Juno" folder exists in that account.
    /// 4. Creates a new note inside the folder with the supplied body.
    /// 5. Returns the note id.
    private static func runCreateScript(body: String, folderName: String?) -> Result<CreatedNote, SinkError> {
        let escapedHTMLBody = Self.escapeForAppleScript(Self.htmlBody(for: body))
        let resolvedFolder = folderName?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? folderName!.trimmingCharacters(in: .whitespacesAndNewlines)
            : JunoNotesFolderName
        let escapedFolder = Self.escapeForAppleScript(resolvedFolder)

        let source = """
        tell application "Notes"
            -- Resolve target account: prefer iCloud, fall back to first account.
            set targetAccount to missing value
            try
                set targetAccount to account "iCloud"
            on error
                if (count of accounts) > 0 then
                    set targetAccount to account 1
                end if
            end try
            if targetAccount is missing value then
                error "no Notes account available"
            end if

            -- Ensure the "Juno" folder exists inside that account.
            set targetFolder to missing value
            tell targetAccount
                if (exists folder "\(escapedFolder)") then
                    set targetFolder to folder "\(escapedFolder)"
                else
                    set targetFolder to make new folder with properties {name:"\(escapedFolder)"}
                end if
            end tell

            -- Notes expects HTML. Juno escapes user text before this script.
            -- Do not set name separately: Notes renders name as the visible
            -- title and also renders body, so using the same first line in
            -- both fields duplicates note text in the app.
            set newNote to make new note at targetFolder with properties {body:"\(escapedHTMLBody)"}
            return id of newNote
        end tell
        """

        guard let script = NSAppleScript(source: source) else {
            return .failure(.scriptError("could not compile AppleScript"))
        }
        var errorInfo: NSDictionary?
        let descriptor = script.executeAndReturnError(&errorInfo)
        if let err = errorInfo {
            let code = (err[NSAppleScript.errorNumber] as? Int) ?? 0
            let msg = (err[NSAppleScript.errorMessage] as? String) ?? "AppleScript error"
            // -1743 = errAEEventNotPermitted (Automation denied).
            // -1744 = errAEEventWouldRequireUserConsent.
            if code == -1743 || code == -1744 {
                return .failure(.automationDenied)
            }
            // -1728 = errAENoSuchObject (often Notes uninstalled or account
            // not yet initialized — first launch on a fresh user).
            if code == -1728 {
                return .failure(.notInstalled)
            }
            return .failure(.scriptError("Notes: \(msg) (\(code))"))
        }
        let identifier = descriptor.stringValue ?? ""
        if identifier.isEmpty {
            // Notes returned a non-string descriptor (object specifier or
            // record). The note was almost certainly created — the make
            // step succeeded — we just can't form a per-note deep link.
            // Log enough to diagnose if this ever shows up in the field
            // and degrade gracefully: hand the user a "notes://" URL so
            // they can still tap "Open in Notes" and find the row in the
            // Juno folder.
            os_log(
                "notes_make_no_identifier descriptor_type=%{public}@",
                log: notesSinkLog, type: .error,
                Self.descriptorTypeName(descriptor)
            )
            return .success(CreatedNote(id: "", url: JunoNotesAppFallbackURL))
        }
        // Notes deep-link: applenotes://showNote?identifier=<numeric tail>.
        // The full id is "x-coredata://…/p123"; the tail after the last "/" is
        // what the URL handler accepts on macOS Sonoma+.
        let tail = identifier.split(separator: "/").last.map(String.init) ?? identifier
        let url = URL(string: "applenotes://showNote?identifier=\(tail)")
            ?? JunoNotesAppFallbackURL
        return .success(CreatedNote(id: identifier, url: url))
    }

    private static func runDeleteScript(id: String) -> Result<Void, SinkError> {
        let escapedId = Self.escapeForAppleScript(id)
        let source = """
        tell application "Notes"
            set targetNote to missing value
            repeat with eachAccount in accounts
                repeat with eachFolder in folders of eachAccount
                    repeat with eachNote in notes of eachFolder
                        if id of eachNote is "\(escapedId)" then
                            set targetNote to eachNote
                            exit repeat
                        end if
                    end repeat
                    if targetNote is not missing value then exit repeat
                end repeat
                if targetNote is not missing value then exit repeat
            end repeat
            if targetNote is missing value then error "note not found"
            delete targetNote
        end tell
        """
        guard let script = NSAppleScript(source: source) else {
            return .failure(.scriptError("could not compile AppleScript"))
        }
        var errorInfo: NSDictionary?
        _ = script.executeAndReturnError(&errorInfo)
        if let err = errorInfo {
            let code = (err[NSAppleScript.errorNumber] as? Int) ?? 0
            let msg = (err[NSAppleScript.errorMessage] as? String) ?? "AppleScript error"
            if code == -1743 || code == -1744 {
                return .failure(.automationDenied)
            }
            return .failure(.scriptError("Notes: \(msg) (\(code))"))
        }
        return .success(())
    }

    /// Human-readable name for an AppleEventDescriptor type code, used
    /// only for logging in the silent-failure path above. Returns
    /// "unknown_<code>" for descriptors we don't have a friendly name
    /// for so the log line is still searchable.
    private static func descriptorTypeName(_ descriptor: NSAppleEventDescriptor) -> String {
        let raw = descriptor.descriptorType
        switch raw {
        case typeUnicodeText: return "unicode_text"
        case typeUTF8Text: return "utf8_text"
        case typeUTF16ExternalRepresentation: return "utf16_text"
        case typeObjectSpecifier: return "object_specifier"
        case typeAERecord: return "ae_record"
        case typeAEList: return "ae_list"
        case typeNull: return "null"
        default:
            // Decode the FourCC for log searchability.
            let bytes: [UInt8] = [
                UInt8((raw >> 24) & 0xff),
                UInt8((raw >> 16) & 0xff),
                UInt8((raw >> 8)  & 0xff),
                UInt8(raw         & 0xff),
            ]
            let four = String(bytes: bytes, encoding: .ascii)?
                .trimmingCharacters(in: .controlCharacters)
            if let four, !four.isEmpty { return "fcc_\(four)" }
            return "unknown_\(raw)"
        }
    }

    private static func escapeForAppleScript(_ s: String) -> String {
        // AppleScript string literals: backslash-escape backslash and quote.
        return s
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    static func htmlBody(for body: String) -> String {
        let escaped = body
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
        let withBreaks = escaped
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .replacingOccurrences(of: "\n", with: "<br>")
        return "<div>\(withBreaks)</div>"
    }

    // MARK: - Signature

    private static let signatureFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "EEE, MMM d 'at' h:mm a"
        return f
    }()

    /// Append "— Captured with Juno · {EEE, MMM d at h:mm a}" on a new line.
    /// Public so unit tests can verify formatting without invoking AppleScript.
    static func applySignature(to body: String, now: Date = Date()) -> String {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        let stamp = signatureFormatter.string(from: now)
        return "\(trimmed)\n\n— Captured with Juno · \(stamp)"
    }
}
