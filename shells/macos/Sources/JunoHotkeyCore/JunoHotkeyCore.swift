import AppKit
import CoreGraphics

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

// MARK: - Fn / Globe key consume policy

/// Pure decision logic for the CGEventTap path that suppresses macOS's emoji
/// picker when Fn is Juno's dictation trigger. Detects bare Fn via
/// ``CGEventFlags.maskSecondaryFn`` only — Apple Silicon may report varying
/// keycodes for the Globe key in ``flagsChanged`` events.
public enum JunoFnGlobeKeyPolicy {
    public static let consumeFnLaunchFlag = "--consume-fn"

    public enum FnEdge: Equatable {
        case none
        case down
        case up
    }

    public struct Decision: Equatable {
        public let edge: FnEdge
        public let consume: Bool

        public static let none = Decision(edge: .none, consume: false)
    }

    public struct Outcome: Equatable {
        public let decision: Decision
        public let fnNowHeld: Bool
    }

    private static let otherModifiers: CGEventFlags = [
        .maskCommand,
        .maskAlternate,
        .maskControl,
        .maskShift,
    ]

    public static func decide(flags: CGEventFlags, fnWasHeld: Bool) -> Outcome {
        let fnDown = flags.contains(.maskSecondaryFn)
        if !fnDown, fnWasHeld {
            let consume = flags.intersection(otherModifiers).isEmpty
            return Outcome(decision: Decision(edge: .up, consume: consume), fnNowHeld: false)
        }
        if !flags.intersection(otherModifiers).isEmpty {
            return Outcome(decision: .none, fnNowHeld: fnWasHeld)
        }
        if fnDown, !fnWasHeld {
            return Outcome(decision: Decision(edge: .down, consume: true), fnNowHeld: true)
        }
        return Outcome(decision: .none, fnNowHeld: fnDown)
    }

    public static func hotkeyLaunchArguments(consumeFn: Bool) -> [String] {
        consumeFn ? [consumeFnLaunchFlag] : []
    }

    public static func stdoutLine(for edge: FnEdge) -> String? {
        switch edge {
        case .none: return nil
        case .down: return "FN_DOWN"
        case .up: return "FN_UP"
        }
    }
}
