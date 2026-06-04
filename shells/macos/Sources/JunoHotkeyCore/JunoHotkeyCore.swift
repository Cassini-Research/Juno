import AppKit

public enum JunoHotkeyEventLine {
    public static let copy = "COPY"
    public static let escape = "ESC"

    private static let cKeyCode: UInt16 = 8
    private static let disallowedCopyModifiers: NSEvent.ModifierFlags = [
        .control,
        .option,
        .shift,
        .function,
    ]

    public static func commandCopyLine(
        keyCode: UInt16,
        modifierFlags: NSEvent.ModifierFlags,
        isRepeat: Bool
    ) -> String? {
        isCommandCopy(keyCode: keyCode, modifierFlags: modifierFlags, isRepeat: isRepeat) ? copy : nil
    }

    public static func isCommandCopy(
        keyCode: UInt16,
        modifierFlags: NSEvent.ModifierFlags,
        isRepeat: Bool
    ) -> Bool {
        guard !isRepeat, keyCode == cKeyCode else { return false }
        let flags = modifierFlags.intersection(.deviceIndependentFlagsMask)
        return flags.contains(.command)
            && flags.intersection(disallowedCopyModifiers).isEmpty
    }

    public static func isCopyLine(_ line: String) -> Bool {
        line == copy
    }
}

public enum JunoCopyReadyShortcutPolicy {
    public static func shouldCopyReadyTranscript(
        hotkeyLine: String,
        copyableTranscript: String?,
        hudStateWire: String
    ) -> Bool {
        JunoHotkeyEventLine.isCopyLine(hotkeyLine)
            && isCopyReady(copyableTranscript: copyableTranscript, hudStateWire: hudStateWire)
    }

    public static func shouldSuppressDictationShortcut(
        copyableTranscript: String?,
        hudStateWire: String
    ) -> Bool {
        isCopyReady(copyableTranscript: copyableTranscript, hudStateWire: hudStateWire)
    }

    private static func isCopyReady(copyableTranscript: String?, hudStateWire: String) -> Bool {
        guard hudStateWire == "idle" else { return false }
        let transcript = copyableTranscript?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return !transcript.isEmpty
    }
}
