// juno-textmon
// AXObserver-based focused-text watcher for the broker's correction-learning
// loop.
//
// Usage:
//   juno-textmon <pid>
//   (first line on stdin = the text the broker just inserted, for baselining)
//
// Stdout protocol:
//   INITIAL:<text>          single-line initial value
//   INITIAL_B64:<base64>    multi-line initial value (base64-encoded UTF-8)
//   CHANGED:<text>          single-line updated value
//   CHANGED_B64:<base64>    multi-line updated value
//   NO_ELEMENT              the focused element could not be resolved
//   NO_VALUE                focused element has no text value attribute
//
// Exit behaviour:
//   - Auto-exits after TIMEOUT_S seconds (bounded lifetime; the broker
//     decides when to start another watch).
//   - Exits cleanly on SIGTERM.
//
// Why we base64 multi-line content:
//   The protocol is line-oriented so the broker can read lines without a
//   framing library. Embedded newlines would break that invariant, so any
//   value containing \n or \r is base64'd.

import Cocoa
import Foundation
import Darwin

let TIMEOUT_S: Double = 30.0
let MAX_VALUE_BYTES = 10_240
let MAX_B64_INPUT   = 7_000

var observedElement: AXUIElement?
var axObserver: AXObserver?
var targetPid: pid_t = 0

func writeOut(_ line: String) {
    FileHandle.standardOutput.write(Data((line + "\n").utf8))
    fflush(stdout)
}

func writeErr(_ line: String) {
    FileHandle.standardError.write(Data((line + "\n").utf8))
}

func emitValue(tag: String, value: String) {
    let truncated = String(value.prefix(MAX_VALUE_BYTES))
    if truncated.contains("\n") || truncated.contains("\r") {
        let capped = String(truncated.prefix(MAX_B64_INPUT))
        let b64 = Data(capped.utf8).base64EncodedString()
        writeOut("\(tag)_B64:\(b64)")
    } else {
        writeOut("\(tag):\(truncated)")
    }
}

func readFocusedValue() -> String? {
    guard let el = observedElement else { return nil }
    var value: AnyObject?
    let rc = AXUIElementCopyAttributeValue(el, kAXValueAttribute as CFString, &value)
    guard rc == .success, let s = value as? String else { return nil }
    return s
}

// AXObserver callback: trailing-refcon style. We don't use refcon; the
// element identity is implicit in `observedElement`.
func axCallback(
    _ observer: AXObserver,
    _ element: AXUIElement,
    _ notification: CFString,
    _ refcon: UnsafeMutableRawPointer?
) {
    if let v = readFocusedValue() {
        emitValue(tag: "CHANGED", value: v)
    }
}

guard CommandLine.arguments.count >= 2,
      let p = Int32(CommandLine.arguments[1]),
      p > 0 else {
    writeErr("usage: juno-textmon <pid>")
    writeOut("NO_ELEMENT")
    exit(1)
}
targetPid = p

// (Informational only — we echo the expected original back so stderr logs
// are self-describing, but we don't use it to drive the observer.)
_ = readLine(strippingNewline: true) ?? ""

let appEl = AXUIElementCreateApplication(targetPid)

var focusedEl: AXUIElement? = nil
for attempt in 1...5 {
    var ref: AnyObject?
    let rc = AXUIElementCopyAttributeValue(
        appEl, kAXFocusedUIElementAttribute as CFString, &ref
    )
    if rc == .success, let e = ref {
        focusedEl = (e as! AXUIElement)
        if attempt > 1 { writeErr("juno-textmon: focused on attempt \(attempt)") }
        break
    }
    writeErr("juno-textmon: attempt \(attempt)/5 failed (code=\(rc.rawValue))")
    if attempt < 5 { Thread.sleep(forTimeInterval: 0.3) }
}

guard let resolved = focusedEl else {
    writeOut("NO_ELEMENT")
    exit(1)
}
observedElement = resolved

guard let initial = readFocusedValue() else {
    writeOut("NO_VALUE")
    exit(0)
}
emitValue(tag: "INITIAL", value: initial)

var createdObs: AXObserver?
let createRc = AXObserverCreate(targetPid, axCallback, &createdObs)
guard createRc == .success, let obs = createdObs else {
    writeErr("juno-textmon: AXObserverCreate rc=\(createRc.rawValue)")
    exit(1)
}
axObserver = obs

let addRc = AXObserverAddNotification(
    obs, resolved, kAXValueChangedNotification as CFString, nil
)
guard addRc == .success else {
    writeErr("juno-textmon: AXObserverAddNotification rc=\(addRc.rawValue)")
    exit(1)
}

CFRunLoopAddSource(
    CFRunLoopGetCurrent(),
    AXObserverGetRunLoopSource(obs),
    .commonModes
)

DispatchQueue.main.asyncAfter(deadline: .now() + TIMEOUT_S) {
    CFRunLoopStop(CFRunLoopGetCurrent())
}

let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
signal(SIGTERM, SIG_IGN)
termSource.setEventHandler { CFRunLoopStop(CFRunLoopGetCurrent()) }
termSource.resume()

CFRunLoopRun()
