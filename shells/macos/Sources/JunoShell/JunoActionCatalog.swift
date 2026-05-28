// JunoActionCatalog.swift
//
// Single source of truth for the Voice Actions feature: every action
// kind, its activate phrases (what the user can say), display copy,
// example utterances for teaching, and the permission descriptor used by
// the Actions page and the Home priority card.
//
// Changing what an action is, what it's called, or what triggers it
// happens **here** — Swift UI, Python parser, and tests all read from
// the same shape. Activate phrases on the Swift side are display-only
// (the parser lives in `juno_core_v3/actions/grammar.py`); keep them in
// sync when adding a new verb.

import Foundation
import SwiftUI

/// Ordered list — drives the Actions page section order and the Home
/// "Try saying" rotation.
let JunoActionCatalogAll: [JunoActionDescriptor] = [
    .reminder, .note, .alarm,
]

struct JunoActionDescriptor: Identifiable, Hashable {
    let kind: JunoActionKind
    let displayName: String
    let pluralName: String
    let symbolName: String
    let accent: Color
    /// One-line value prop for cards.
    let blurb: String
    /// What the user can say (display-only — parser is authoritative).
    let activatePhrases: [String]
    /// 2–3 example utterances shown as quote chips on the Actions page.
    let examples: [String]
    /// Permission descriptor for the unified status row.
    let permission: JunoActionPermissionDescriptor

    var id: JunoActionKind { kind }
}

extension JunoActionDescriptor {
    static let reminder = JunoActionDescriptor(
        kind: .reminder,
        displayName: "Reminder",
        pluralName: "Reminders",
        symbolName: "bell.badge",
        accent: Color(red: 0.95, green: 0.55, blue: 0.20),
        blurb: "Time-based pings that ring through Apple Reminders.",
        activatePhrases: [
            "remind me",
            "reminder",
            "set a reminder",
            "remember to",
            "remember that",
        ],
        examples: [
            "Hey Juno, remind me to call mom at 6pm",
            "Juno reminder: stand-up tomorrow at 9",
            "Juno, remember to pay rent on the 1st",
        ],
        permission: .reminders
    )

    static let note = JunoActionDescriptor(
        kind: .note,
        displayName: "Note",
        pluralName: "Notes",
        symbolName: "note.text",
        accent: Color(red: 0.95, green: 0.78, blue: 0.30),
        blurb: "Saved to a folder called \u{201C}Juno\u{201D} in Apple Notes.",
        activatePhrases: [
            "take a note",
            "make a note",
            "note this",
            "note that",
            "save this note",
            "jot this down",
            "write this down",
        ],
        examples: [
            "Hey Juno, take a note: action items from the standup are…",
            "Juno, note that the new API key rotates on Mondays",
            "Juno jot this down — book idea: a memoir told in voicemails",
        ],
        permission: .notesAutomation
    )

    static let alarm = JunoActionDescriptor(
        kind: .alarm,
        displayName: "Alarm",
        pluralName: "Alarms",
        symbolName: "alarm",
        accent: Color(red: 0.55, green: 0.45, blue: 0.95),
        blurb: "Saved as one-shot Calendar alerts so they ring even when Juno is closed.",
        activatePhrases: [
            "set an alarm",
            "alarm",
            "wake me",
            "wake me up",
            "set a timer for",
            "ping me",
        ],
        examples: [
            "Hey Juno, set an alarm for 7am tomorrow",
            "Juno, wake me up in 25 minutes",
            "Juno alarm at 3:30 to leave for the airport",
        ],
        permission: .calendarEvents
    )
}

/// Sugar that lets callers write `JunoActionKind.reminder.descriptor`.
extension JunoActionKind {
    var descriptor: JunoActionDescriptor {
        switch self {
        case .reminder: return .reminder
        case .note: return .note
        case .alarm: return .alarm
        }
    }
}
