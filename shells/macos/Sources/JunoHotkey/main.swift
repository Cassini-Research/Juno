// juno-hotkey
// Global listener for modifier-only and Fn/Globe hotkeys.
//
// Why this exists separately from JunoShell:
//   - NSEvent global monitors must run on an active NSApplication, but the
//     menu-bar app should not be coupled to the hotkey run loop. Running the
//     listener as its own accessory process lets us restart one without
//     disturbing the other, and keeps the hotkey working during menu-bar UI
//     work (ex. settings window rebuild).
//
// Stdout protocol (one event per line, UTF-8):
//   FN_DOWN                   — Globe/Fn pressed
//   FN_UP                     — Globe/Fn released
//   RIGHT_MOD_DOWN:<name>     — right-side Command/Control/Option/Shift pressed
//   RIGHT_MOD_UP:<name>       — ...released
//   MODIFIER_UP:<name>        — any modifier in our watch set went from held -> not held
//   OPT_SPACE_DOWN / OPT_SPACE_UP   — Option + Space (hold-to-talk chord)
//   CTRL_SPACE_DOWN / CTRL_SPACE_UP — Control + Space
//   ESC                             — Escape (HUD dismiss / cancel dictation)
//   COPY                            — Command + C
//   HOTKEY_DEGRADED:<which>          — a monitor failed to install
//
// When launched with ``--consume-fn``, bare Fn/Globe ``flagsChanged`` events are
// consumed via a session-level CGEventTap so macOS does not open the emoji
// picker. JunoShell passes that flag only when Fn is the user's dictation key.

import Cocoa
import CoreGraphics
import Darwin
import JunoHotkeyCore

var fnHeld = false
var lastMods: NSEvent.ModifierFlags = []

// Right-side modifier key codes are stable kVK_Right* constants from
// <HIToolbox/Events.h>. We match them here without importing HIToolbox.
let rightModifiers: [(UInt16, NSEvent.ModifierFlags, String)] = [
    (61, .option, "RightOption"),
    (54, .command, "RightCommand"),
    (62, .control, "RightControl"),
    (60, .shift, "RightShift"),
]
let watchMask: NSEvent.ModifierFlags = [.control, .command, .option, .shift]
let watchNames: [(NSEvent.ModifierFlags, String)] = [
    (.control, "control"),
    (.command, "command"),
    (.option, "option"),
    (.shift, "shift"),
]

@inline(__always)
func emit(_ line: String) {
    FileHandle.standardOutput.write(Data((line + "\n").utf8))
    fflush(stdout)
}

var optSpaceDown = false
var ctrlSpaceDown = false

// When true, Fn is handled by the consuming CGEventTap below instead of the
// passive NSEvent flags monitor.
var consumeFnViaEventTap = CommandLine.arguments.contains(JunoFnGlobeKeyPolicy.consumeFnLaunchFlag)
var fnTapHeld = false
var fnEventTap: CFMachPort?
var fnTapReEnableAttempts = 0
let maxFnTapReEnableAttempts = 3

func setupFnConsumeEventTap() -> Bool {
    let eventMask = CGEventMask(1 << CGEventType.flagsChanged.rawValue)

    let callback: CGEventTapCallBack = { _, type, event, _ in
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            fnTapReEnableAttempts += 1
            if fnTapReEnableAttempts <= maxFnTapReEnableAttempts, let tap = fnEventTap {
                CGEvent.tapEnable(tap: tap, enable: true)
            }
            return Unmanaged.passUnretained(event)
        }

        fnTapReEnableAttempts = 0

        guard type == .flagsChanged else {
            return Unmanaged.passUnretained(event)
        }

        let outcome = JunoFnGlobeKeyPolicy.decide(flags: event.flags, fnWasHeld: fnTapHeld)
        fnTapHeld = outcome.fnNowHeld

        if let line = JunoFnGlobeKeyPolicy.stdoutLine(for: outcome.decision.edge) {
            emit(line)
        }

        if outcome.decision.consume {
            return nil
        }
        return Unmanaged.passUnretained(event)
    }

    guard let tap = CGEvent.tapCreate(
        tap: .cgSessionEventTap,
        place: .headInsertEventTap,
        options: .defaultTap,
        eventsOfInterest: eventMask,
        callback: callback,
        userInfo: nil
    ) else {
        return false
    }

    fnEventTap = tap
    guard let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0) else {
        return false
    }
    CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
    CGEvent.tapEnable(tap: tap, enable: true)
    return true
}

if consumeFnViaEventTap {
    if setupFnConsumeEventTap() {
        fnHeld = false
    } else {
        FileHandle.standardError.write(
            Data("juno-hotkey: Fn consume tap unavailable — falling back to passive Fn monitor (Input Monitoring / Accessibility may be required).\n".utf8)
        )
        emit("HOTKEY_DEGRADED:fntap")
        consumeFnViaEventTap = false
    }
}

let monitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { event in
    let flags = event.modifierFlags

    if !consumeFnViaEventTap {
        let fnNow = flags.contains(.function)
        if fnNow && !fnHeld {
            fnHeld = true
            emit("FN_DOWN")
        } else if !fnNow && fnHeld {
            fnHeld = false
            emit("FN_UP")
        }
    }

    let keyCode = event.keyCode
    for (code, flag, name) in rightModifiers where keyCode == code {
        emit(flags.contains(flag) ? "RIGHT_MOD_DOWN:\(name)" : "RIGHT_MOD_UP:\(name)")
        break
    }

    let now = flags.intersection(watchMask)
    if now != lastMods {
        let released = lastMods.subtracting(now)
        for (flag, name) in watchNames where released.contains(flag) {
            emit("MODIFIER_UP:\(name)")
        }
        lastMods = now
    }
}

let keyMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.keyDown, .keyUp]) { event in
    if event.type == .keyDown,
       let line = JunoHotkeyEventLine.commandCopyLine(
        keyCode: event.keyCode,
        modifierFlags: event.modifierFlags,
        isRepeat: event.isARepeat
       ) {
        emit(line)
        return
    }

    guard event.keyCode == 49 else { return }
    let f = event.modifierFlags
    switch event.type {
    case .keyDown:
        if event.isARepeat { return }
        if f.contains(.command) { return }
        if f.contains(.option), !f.contains(.control) {
            if !optSpaceDown {
                optSpaceDown = true
                emit("OPT_SPACE_DOWN")
            }
        } else if f.contains(.control), !f.contains(.option) {
            if !ctrlSpaceDown {
                ctrlSpaceDown = true
                emit("CTRL_SPACE_DOWN")
            }
        }
    case .keyUp:
        if optSpaceDown {
            optSpaceDown = false
            emit("OPT_SPACE_UP")
        }
        if ctrlSpaceDown {
            ctrlSpaceDown = false
            emit("CTRL_SPACE_UP")
        }
    default:
        break
    }
}

if keyMonitor == nil {
    FileHandle.standardError.write(
        Data("juno-hotkey: key monitor unavailable — Option/Ctrl+Space chords and Command+C disabled (Input Monitoring / Accessibility may be required).\n".utf8)
    )
    emit("HOTKEY_DEGRADED:key")
}

let escMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { event in
    guard event.keyCode == 53 else { return }
    if event.isARepeat { return }
    emit(JunoHotkeyEventLine.escape)
}

if escMonitor == nil {
    FileHandle.standardError.write(
        Data("juno-hotkey: Esc monitor unavailable — HUD Escape-to-cancel disabled (Input Monitoring / Accessibility may be required).\n".utf8)
    )
    emit("HOTKEY_DEGRADED:esc")
}

guard monitor != nil else {
    FileHandle.standardError.write(Data("juno-hotkey: global flags monitor install failed\n".utf8))
    emit("HOTKEY_DEGRADED:flags")
    exit(1)
}

// Clean shutdown on SIGTERM/SIGINT so launchd-style supervisors see exit 0.
let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
let intSource  = DispatchSource.makeSignalSource(signal: SIGINT,  queue: .main)
signal(SIGTERM, SIG_IGN)
signal(SIGINT,  SIG_IGN)
for src in [termSource, intSource] {
    src.setEventHandler { exit(0) }
    src.resume()
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
app.run()
