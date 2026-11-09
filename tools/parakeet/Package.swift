// swift-tools-version: 6.0
// Parakeet TDT speech recognition and Kokoro text-to-speech (FluidAudio, Core ML on the Neural Engine) as
// stdin/stdout processes.
import PackageDescription

let package = Package(
    name: "parakeet-serve",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.18.2"),
    ],
    targets: [
        .executableTarget(
            name: "parakeet-serve",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")],
            path: "Sources/parakeet-serve"
        ),
        .executableTarget(
            name: "kokoro-serve",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")],
            path: "Sources/kokoro-serve"
        ),
    ]
)
