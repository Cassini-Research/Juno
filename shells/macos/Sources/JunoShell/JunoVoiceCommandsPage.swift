// JunoVoiceCommandsPage.swift
//
// "Voice Commands" sidebar destination. These are the editing /
// transform phrases Juno recognises during dictation. They do not
// require Reminders, Calendar, or Notes permissions — they're pure
// in-flight edits to what you're saying or what you just said.
//
// Layout follows System Settings → Keyboard → Shortcuts: grouped
// rows with the phrase on the left and a plain-English effect on
// the right. No chip pills, no rotating examples, no dev-tool feel.
//
// Commands that hand off to the on-device writer (Qwen) are marked
// with a subtle "Writer" tag so the user understands which phrases
// take a moment longer than the deterministic ones.

import SwiftUI

struct JunoVoiceCommandsPage: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 18) {
                header
                howItWorksCard
                groupCard(
                    title: "While you're talking",
                    footnote: "Instant — recognised the moment you say them.",
                    rows: kInMomentVoiceCommands
                )
                groupCard(
                    title: "Edit what you just said",
                    footnote: "Operates on the last sentence — or your selection, if you have text highlighted when Juno starts.",
                    rows: kRecentEditVoiceCommands
                )
                writerNoteCard
            }
            .junoDetailPagePadding()
        }
    }

    // MARK: - Header

    private var header: some View {
        JunoPageHeader(
            eyebrow: "Voice",
            title: "Voice Commands",
            subtitle: "Edit and reshape your dictation without lifting your hands. Say them as part of what you're already saying — no wake phrase required.",
            trailing: { EmptyView() }
        )
    }

    // MARK: - How it works

    private var howItWorksCard: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "waveform")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(JunoDesignTokens.accent)
                .frame(width: 22)
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 4) {
                Text("Speak naturally")
                    .junoType(.bodyEmphasis)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("Voice Commands run during dictation. They don't need any macOS permissions — they only edit text Juno is about to type, or text Juno just typed. If you have text selected in the app, every \u{201C}edit what you just said\u{201D} command operates on the selection instead.")
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .premiumCard()
    }

    // MARK: - Command group card

    private func groupCard(title: String, footnote: String, rows: [VoiceCommandRow]) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(title)
                .junoType(.bodyEmphasis)
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .padding(.horizontal, 14)
                .padding(.top, 14)
                .padding(.bottom, 6)

            Text(footnote)
                .junoType(.caption)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .padding(.horizontal, 14)
                .padding(.bottom, 10)
                .fixedSize(horizontal: false, vertical: true)

            VStack(spacing: 0) {
                ForEach(Array(rows.enumerated()), id: \.element.id) { idx, row in
                    voiceCommandRow(row)
                    if idx < rows.count - 1 {
                        Divider()
                            .opacity(0.35)
                            .padding(.leading, 14)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .premiumCard()
    }

    private func voiceCommandRow(_ row: VoiceCommandRow) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text("\u{201C}" + row.phrase + "\u{201D}")
                .font(.system(size: 13, weight: .medium, design: .rounded))
                .foregroundStyle(JunoTheme.primaryText(scheme))
                .frame(minWidth: 200, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
            Text(row.effect)
                .junoType(.caption)
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
            if row.usesWriter {
                writerTag
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
    }

    private var writerTag: some View {
        Text("WRITER")
            .font(.system(size: 9, weight: .semibold, design: .monospaced))
            .tracking(0.8)
            .foregroundStyle(JunoTheme.secondaryText(scheme))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .overlay(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .stroke(JunoTheme.secondaryText(scheme).opacity(0.35), lineWidth: 0.6)
            )
    }

    private var writerNoteCard: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "sparkles")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(JunoTheme.secondaryText(scheme))
                .frame(width: 22)
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 3) {
                Text("About the WRITER tag")
                    .junoType(.bodyEmphasis)
                    .foregroundStyle(JunoTheme.primaryText(scheme))
                Text("These commands run through Juno's on-device writer model. They take a moment longer than the instant edits — usually under a second — and run entirely on your Mac.")
                    .junoType(.caption)
                    .foregroundStyle(JunoTheme.secondaryText(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .premiumCard()
    }
}

// MARK: - Command data

private struct VoiceCommandRow: Identifiable {
    let phrase: String
    let effect: String
    let usesWriter: Bool
    var id: String { phrase }
}

/// In-moment commands. These are deterministic — pure text manipulation
/// or insertion that doesn't touch the writer model. Recognised the
/// instant the phrase finishes.
private let kInMomentVoiceCommands: [VoiceCommandRow] = [
    .init(phrase: "scratch that", effect: "Erase the sentence you're saying right now.", usesWriter: false),
    .init(phrase: "undo that", effect: "Undo Juno's last typed change.", usesWriter: false),
    .init(phrase: "delete that", effect: "Delete the last thing Juno typed.", usesWriter: false),
    .init(phrase: "delete last word", effect: "Erase the most recent word.", usesWriter: false),
    .init(phrase: "delete last two words", effect: "Erase the last two words.", usesWriter: false),
    .init(phrase: "delete last sentence", effect: "Erase the last full sentence.", usesWriter: false),
    .init(phrase: "new line", effect: "Insert a line break.", usesWriter: false),
    .init(phrase: "new paragraph", effect: "Insert a paragraph break.", usesWriter: false),
    .init(phrase: "next bullet", effect: "Start the next bulleted item.", usesWriter: false),
    .init(phrase: "next number", effect: "Start the next numbered item.", usesWriter: false),
    .init(phrase: "open quote", effect: "Insert an opening quotation mark.", usesWriter: false),
    .init(phrase: "close quote", effect: "Insert a closing quotation mark.", usesWriter: false),
]

/// Recent-edit / transform commands. Most of these rewrite text through
/// the on-device writer. The two "delete the last …" entries are
/// deterministic, like their in-moment cousins.
private let kRecentEditVoiceCommands: [VoiceCommandRow] = [
    .init(phrase: "fix that", effect: "Clean up the last sentence — grammar, punctuation, light edits.", usesWriter: true),
    .init(phrase: "make that shorter", effect: "Tighten the last sentence to its core meaning.", usesWriter: true),
    .init(phrase: "make that longer", effect: "Expand the last sentence with more detail.", usesWriter: true),
    .init(phrase: "make that clearer", effect: "Rewrite the last sentence to read more cleanly.", usesWriter: true),
    .init(phrase: "make that more formal", effect: "Lift the tone of the last sentence.", usesWriter: true),
    .init(phrase: "make that more casual", effect: "Loosen the tone of the last sentence.", usesWriter: true),
    .init(phrase: "turn that into bullets", effect: "Rewrite the last passage as a bulleted list.", usesWriter: true),
    .init(phrase: "turn that into a numbered list", effect: "Rewrite the last passage as a numbered list.", usesWriter: true),
    .init(phrase: "translate that to \u{2039}language\u{203A}", effect: "Translate the last sentence into the language you name.", usesWriter: true),
    .init(phrase: "replace \u{2039}word\u{203A} with \u{2039}word\u{203A}", effect: "Swap a specific word in the last sentence.", usesWriter: true),
    .init(phrase: "delete the last sentence", effect: "Remove the most recent full sentence.", usesWriter: false),
    .init(phrase: "delete the last paragraph", effect: "Remove the most recent paragraph.", usesWriter: false),
]
