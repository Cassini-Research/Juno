import Foundation

/// User-facing copy for the menu-bar "Voice engine" help surface. Single
/// source of truth so naming-discipline regressions can be caught with one
/// regex (see JunoBrokerHelpCopyTests).
enum JunoEngineHelpCopy {
    static let helpWindowTitle = "Juno · Voice engine"
    static let helpWindowSubtitle = "Local dictation — runs only on your Mac"

    static let headlineConnected = "Voice engine is running"
    static let headlineDisconnected = "Voice engine isn’t connected"

    static let connectedExplanation =
        "Juno can transcribe on this Mac. You’re set as soon as microphone and Accessibility are allowed."

    static let disconnectedExplanation =
        "Juno’s voice engine runs only on this Mac — nothing is sent to the cloud. It starts automatically when you open Juno from your Applications folder."

    static let checkAgainButton = "Check again"

    static let tryThisFirstHeading = "Try this first"
    static let tryThisFirstBody =
        "From the menu bar, choose Quit Juno, then open Juno from your Applications folder and try again."

    static let stillStuckHeading = "Still stuck?"
    static let stillStuckBody =
        "If it still doesn’t connect, reinstall Juno from your download link."

    /// Concatenation used by the naming-discipline test. Every user-visible
    /// string in the help surface must appear here.
    static let allUserFacingStrings: [String] = [
        helpWindowTitle,
        helpWindowSubtitle,
        headlineConnected,
        headlineDisconnected,
        connectedExplanation,
        disconnectedExplanation,
        checkAgainButton,
        tryThisFirstHeading,
        tryThisFirstBody,
        stillStuckHeading,
        stillStuckBody,
    ]
}

/// User-facing copy for engine reachability errors. Lives here so naming
/// discipline can be policed from a single source.
enum JunoEngineErrorCopy {
    /// Shown when something is on the engine port but isn't the bundled
    /// runtime (e.g. a developer workbench, or another local app squatting
    /// the port). Must avoid Python / shell / "workbench" / Ctrl+C terms.
    static let roleMismatchUserMessage =
        "Another local Juno service is already running on this Mac. Quit it from its menu bar (or the app it came from) and reopen Juno."

    /// Shown when the running engine reports a non-production deployment
    /// profile. Must avoid raw backend names (preview=…, final=…, etc.).
    static let profileMismatchUserMessage =
        "Juno's voice engine is running an outdated configuration. Quit and reopen Juno to refresh it."
}

/// Maps internal classifier / mode ids to short, non-technical UI strings.
enum JunoUserFacingCopy {
    /// Human label for `classify_app_category` values; returns `nil` for unknown / empty.
    static func appWritingContext(raw: String?) -> String? {
        let key = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if key.isEmpty || key == "unknown" { return nil }
        switch key {
        case "messaging": return "Messages & chat"
        case "email": return "Email"
        case "docs": return "Notes & documents"
        case "code": return "Code editor"
        case "terminal": return "Terminal"
        case "forms": return "Forms & fields"
        case "meeting": return "Meeting notes"
        default:
            return key.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    /// Short title for built-in writer mode ids from the broker.
    static func builtinModeTitle(id: String) -> String {
        let key = id.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch key {
        case "default_surface": return "Default"
        case "verbatim": return "Verbatim"
        case "casual_chat": return "Casual chat"
        case "formal_email": return "Formal email"
        case "structured_notes": return "Structured notes"
        case "explicit_rewrite": return "Polished rewrite"
        case "command_mode": return "Commands"
        default:
            return id.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    /// Longer guidance for the Styles detail pane — specific to each built-in id.
    static func builtinModeDetail(id: String) -> String {
        let key = id.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch key {
        case "default_surface":
            return "Juno picks tone and light cleanup based on the app you are in—chat stays punchy, email stays polished. Best everyday choice when you do not want to micromanage a style. Activate it when you switch contexts often and still want consistent quality."
        case "verbatim":
            return "Keeps your words as close to what you said as possible—ideal for quotes, legal phrasing, code-adjacent wording, or any time rewriting would be wrong. Expect fewer automatic fixes; you trade polish for fidelity."
        case "casual_chat":
            return "Light cleanup for Slack, Messages, and similar—conversational tone, fewer stiff formalities. Use when you want dictation to feel like natural chat, not a memo. Not the best pick for client-facing email where you need full polish."
        case "formal_email":
            return "Biases toward complete sentences, clearer structure, and professional tone—suited to Mail, Gmail, and long-form messages. Use when the reader expects polish. Switch away for quick internal chat where formality slows you down."
        case "structured_notes":
            return "Encourages headings, bullets, and skimmable structure for notes, specs, and meeting capture. Use when you will revisit the text later and need organization. Less ideal for one-line replies where structure adds noise."
        case "explicit_rewrite":
            return "Allows a stronger pass for clarity and flow—best when you want the text to read smoothly even if it moves further from raw dictation. Use for drafts you will edit lightly afterward; avoid when every word must stay exact."
        case "command_mode":
            return "Optimizes for voice commands that control Juno (open settings, switch style, insert snippets, etc.) instead of open-ended dictation. Use when you are driving the app by voice; switch back to a writing style for normal transcription."
        default:
            return "This built-in style adjusts formatting, commands, and multilingual behavior. Activate it to make it the default for new dictation sessions, or create a custom style on top of it for your own instructions."
        }
    }
}
