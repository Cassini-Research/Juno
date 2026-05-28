// juno-paste
// Post Cmd+V into the frontmost app via CGEvent.
//
// Requires the process bundle to have Accessibility permission. Exit codes:
//   0  — success
//   2  — not AX-trusted; caller should prompt the user
//   3  — could not create CGEvent (rare; low-memory or session issues)
//
// Design notes:
//   - We deliberately synthesize keyDown and keyUp *separately* with a small
//     delay, because several Electron apps (Slack, Notion) drop synthetic
//     Cmd+V when the events arrive back-to-back within one run-loop tick.
//   - `.cgSessionEventTap` is used so the event is injected at the session
//     level and is visible to every app (vs `.cgAnnotatedSessionEventTap`
//     which some sandboxed apps treat differently).
//   - We use ``CGEventSource(stateID: .privateState)`` so the synthetic
//     Cmd+V's modifier state is INDEPENDENT of the physical keyboard. This
//     matters for the broker-on-pause flow: the user is typically still
//     holding the dictation hotkey (often Option) when a pause-snapshot
//     returns and we synthesise Cmd+V. With the default combined-session
//     source the OS merged the physical Option with the synthetic Cmd
//     and the receiver saw ``Cmd+Option+V`` — which Notes / TextEdit /
//     most apps treat as "Paste and Match Style" (different action) or
//     ignore entirely. A private-state source has its own modifier
//     accumulator, so the receiver sees exactly the ``flags`` we set
//     (``.maskCommand``) regardless of what's physically held down.

import Cocoa

guard AXIsProcessTrusted() else {
    FileHandle.standardError.write(Data("juno-paste: accessibility not trusted\n".utf8))
    exit(2)
}

let vKey: CGKeyCode = 0x09  // 'v'

let source = CGEventSource(stateID: .privateState)

guard
    let down = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: true),
    let up   = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: false)
else {
    FileHandle.standardError.write(Data("juno-paste: CGEvent creation failed\n".utf8))
    exit(3)
}

down.flags = .maskCommand
up.flags   = .maskCommand

down.post(tap: .cgSessionEventTap)
usleep(8_000)   // 8ms — matches what most user-space paste helpers use
up.post(tap: .cgSessionEventTap)
usleep(20_000)  // let the receiving app flush the paste
exit(0)
