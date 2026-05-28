// juno-host
//
// One-shot host-resource probe. Prints a single JSON line describing
// the machine's current thermal / battery / memory state so the broker
// can degrade aggressive modes (streaming ASR, writer LLM transforms)
// before the user notices a stutter.
//
// Stdout protocol:
//
//   {
//     "thermal_pressure": "nominal" | "warning" | "critical",
//     "battery_low": false,
//     "memory_pressure":  "nominal" | "warning" | "critical"
//   }
//
// Why a dedicated helper (rather than extending juno-capability):
//   juno-capability answers "is dictation safe in this app right now?"
//   (secure-field, blocklist). juno-host answers "is the machine OK to
//   run big models right now?". Different cadence, different failure
//   modes, different privilege surface (host probe needs no AX trust).
//   Keeping them separate lets the broker poll each at a different
//   frequency and keeps the "dictation was blocked" and "dictation is
//   running degraded" UX decoupled.
//
// Exit code:
//   0 always. Failures surface as "unknown" values in the JSON; the
//   broker treats them as "no signal" and stays in its default
//   (unconstrained) policy.

import Darwin
import Foundation
import IOKit
import IOKit.ps

// MARK: - Thermal pressure

/// Maps ``ProcessInfo.ThermalState`` onto the broker's coarse
/// pressure bucket. We intentionally collapse ``.serious`` and
/// ``.critical`` onto the same bucket ("critical") because by the
/// time macOS reports either, running a ~400 MB whisper model is a
/// bad idea — the broker's degrade policy doesn't need finer
/// granularity and we don't want to force callers to memorize
/// Apple's four-level enum.
private func probeThermalPressure() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "nominal"
    case .serious: return "warning"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

// MARK: - Battery

/// Returns (battery_low, on_power).
/// ``battery_low`` is true when battery is the *current* power source
/// AND ``kIOPSCurrentCapacityKey`` is < 20% of ``kIOPSMaxCapacityKey``.
/// Desktops (Mac mini / Studio / Pro) never return true here because
/// the "InternalBattery" source is absent — ``IOPSCopyPowerSourcesInfo``
/// returns an empty sources array.
private func probeBatteryLow() -> Bool {
    guard let snapshot = IOPSCopyPowerSourcesInfo() else {
        return false
    }
    let info = snapshot.takeRetainedValue()
    guard let sources = IOPSCopyPowerSourcesList(info)?.takeRetainedValue() as? [CFTypeRef] else {
        return false
    }
    for src in sources {
        guard let description = IOPSGetPowerSourceDescription(info, src)?.takeUnretainedValue()
                as? [String: Any] else { continue }
        // Only the internal battery counts; external (UPS, accessory)
        // batteries produce noisy "low" signals that would blanket-
        // degrade the broker even when the user is plugged into wall
        // power.
        guard let type = description[kIOPSTypeKey] as? String,
              type == kIOPSInternalBatteryType else { continue }
        let state = description[kIOPSPowerSourceStateKey] as? String ?? ""
        let onBattery = state == kIOPSBatteryPowerValue
        let capacity = description[kIOPSCurrentCapacityKey] as? Int ?? 100
        let max = description[kIOPSMaxCapacityKey] as? Int ?? 100
        if max > 0, onBattery, (capacity * 100 / max) < 20 {
            return true
        }
    }
    return false
}

// MARK: - Memory pressure

/// Estimates memory pressure by looking at free vs wired+active pages.
/// The Apple-recommended API (``kern.memorystatus_level`` sysctl) is
/// gated on private entitlements; we stay user-space by computing a
/// free-fraction from ``host_statistics64(HOST_VM_INFO64)``.
///
/// Buckets:
///   - nominal:  free >= 20% OR we couldn't read stats
///   - warning:  10% <= free < 20%
///   - critical: free < 10%
private func probeMemoryPressure() -> String {
    var info = vm_statistics64_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
    let host = mach_host_self()
    let kr: kern_return_t = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
            host_statistics64(host, HOST_VM_INFO64, rebound, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return "unknown" }
    let pageSize = UInt64(vm_kernel_page_size)
    let free = UInt64(info.free_count) * pageSize
    let active = UInt64(info.active_count) * pageSize
    let inactive = UInt64(info.inactive_count) * pageSize
    let wired = UInt64(info.wire_count) * pageSize
    let compressed = UInt64(info.compressor_page_count) * pageSize
    let total = free + active + inactive + wired + compressed
    guard total > 0 else { return "unknown" }
    let freeFrac = Double(free + inactive) / Double(total)
    if freeFrac < 0.10 { return "critical" }
    if freeFrac < 0.20 { return "warning" }
    return "nominal"
}

// MARK: - Main

let payload: [String: Any] = [
    "thermal_pressure": probeThermalPressure(),
    "battery_low": probeBatteryLow(),
    "memory_pressure": probeMemoryPressure(),
]

do {
    let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
} catch {
    let fallback = #"{"battery_low":false,"error":"json_serialize_failed","memory_pressure":"unknown","thermal_pressure":"unknown"}"#
    FileHandle.standardError.write(Data("juno-host: JSON serialization failed: \(error)\n".utf8))
    FileHandle.standardOutput.write(Data(fallback.utf8))
}
exit(0)
