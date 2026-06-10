// swift-tools-version: 5.9
// Juno macOS shell — desktop surface for the broker plane.
//
// Targets:
//   - Juno (product) : SwiftUI menu-bar app. Talks to the broker over HTTP.
//   - juno-paste     : posts Cmd+V via CGEvent into the frontmost app.
//   - juno-hotkey    : global Fn (Globe) + right-side modifier listener; emits
//                      DOWN/UP lines on stdout. Used by Juno to trigger
//                      push-to-talk without owning the main run loop.
//   - juno-textmon   : AXObserver that watches the focused-field value changes
//                      of a given PID; streams INITIAL/CHANGED lines on stdout
//                      so the broker can learn from post-paste corrections.
//   - juno-capability: one-shot Accessibility probe. Prints a JSON line
//                      describing the frontmost app + focused UI element
//                      (including secure-field detection). The broker uses
//                      this to refuse dictation into password fields or
//                      blocklisted apps before the mic even opens.
//   - juno-host      : one-shot host-resource probe. Prints thermal,
//                      battery, and memory-pressure buckets the broker
//                      uses to decide whether to degrade streaming or
//                      writer model transforms on a hot/under-powered
//                      machine.
//
// Build: swift build -c release
// Run:   .build/release/Juno (expects broker/workbench on :8765)

import PackageDescription

let package = Package(
    name: "JunoShell",
    platforms: [.macOS("15.0")],
    products: [
        .executable(name: "Juno", targets: ["JunoShell"]),
        .executable(name: "juno-paste", targets: ["JunoPaste"]),
        .executable(name: "juno-hotkey", targets: ["JunoHotkey"]),
        .executable(name: "juno-textmon", targets: ["JunoTextMonitor"]),
        .executable(name: "juno-capability", targets: ["JunoCapability"]),
        .executable(name: "juno-host", targets: ["JunoHost"]),
    ],
    dependencies: [
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.9.1"),
    ],
    targets: [
        .executableTarget(
            name: "JunoShell",
            dependencies: [
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources/JunoShell",
            exclude: ["JunoShellInfo.plist"],
            resources: [
                .process("Resources"),
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Sources/JunoShell/JunoShellInfo.plist",
                    "-Xlinker", "-rpath",
                    "-Xlinker", "@executable_path/../Frameworks",
                ]),
            ]
        ),
        .executableTarget(
            name: "JunoPaste",
            path: "Sources/JunoPaste"
        ),
        .executableTarget(
            name: "JunoHotkey",
            path: "Sources/JunoHotkey"
        ),
        .executableTarget(
            name: "JunoTextMonitor",
            path: "Sources/JunoTextMonitor"
        ),
        .executableTarget(
            name: "JunoCapability",
            path: "Sources/JunoCapability"
        ),
        .executableTarget(
            name: "JunoHost",
            path: "Sources/JunoHost"
        ),
    ]
)
