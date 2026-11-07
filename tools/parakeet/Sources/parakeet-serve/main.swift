// parakeet-serve: Parakeet TDT 0.6B v2 (English) speech recognition through FluidAudio, Core ML on the
// Neural Engine. One long-running process; audio arrives on stdin, transcripts leave on stdout, so no
// audio touches disk and no network is needed (models are loaded from a local folder).
//
//   parakeet-serve --models DIR --download      fetch the Core ML bundles into DIR once (needs network)
//   parakeet-serve --models DIR [--encoder ane|gpu]
//
// Protocol (serve mode):
//   request:  UInt32 little-endian sample count n, then n Float32 little-endian samples (16 kHz mono).
//             n == 0 ends the process.
//   response: one line "@@" + JSON: {"text", "seconds", "audio_s", "confidence"} or {"error"}.
//   startup:  "@@" + {"ready": true, "load_s", "warm_s"} once models are loaded and warm.
// Lines without the "@@" prefix are library logging and should be ignored.

import CoreML
import FluidAudio
import Foundation

let MIN_SAMPLES = 8000   // FluidAudio rejects clips under 0.3 s; pad short clips with silence to 0.5 s

func emit(_ obj: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: obj)
    FileHandle.standardOutput.write("@@".data(using: .utf8)! + data + "\n".data(using: .utf8)!)
}

func readExactly(_ n: Int) -> Data? {
    var out = Data()
    while out.count < n {
        let chunk = FileHandle.standardInput.readData(ofLength: n - out.count)
        if chunk.isEmpty { return nil }   // EOF
        out.append(chunk)
    }
    return out
}

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}

guard let modelsPath = arg("--models") else {
    FileHandle.standardError.write("usage: parakeet-serve --models DIR [--download] [--encoder ane|gpu]\n".data(using: .utf8)!)
    exit(2)
}
let modelsDir = URL(fileURLWithPath: modelsPath, isDirectory: true)

if CommandLine.arguments.contains("--download") {
    do {
        let dir = try await AsrModels.download(to: modelsDir, version: .v2)
        print("downloaded to \(dir.path)")
        exit(0)
    } catch {
        FileHandle.standardError.write("download failed: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}

let encoderUnits: MLComputeUnits? = (arg("--encoder") == "gpu") ? .cpuAndGPU : nil   // nil = Neural Engine

let asr: AsrManager
var t0 = Date()
do {
    let models = try await AsrModels.load(from: modelsDir, version: .v2, encoderComputeUnits: encoderUnits)
    asr = AsrManager(config: .default)
    try await asr.loadModels(models)
} catch {
    emit(["error": "load failed: \(error)"])
    exit(1)
}
let loadS = Date().timeIntervalSince(t0)

let layers = await asr.decoderLayerCount

func transcribe(_ samples: [Float]) async throws -> ASRResult {
    var s = samples
    if s.count < MIN_SAMPLES { s += [Float](repeating: 0, count: MIN_SAMPLES - s.count) }
    var state = TdtDecoderState.make(decoderLayers: layers)
    return try await asr.transcribe(s, decoderState: &state)
}

// Warm the Core ML programs (first prediction compiles/loads them on the Neural Engine).
t0 = Date()
var noise = [Float](repeating: 0, count: 16000)
for i in noise.indices { noise[i] = Float.random(in: -0.01...0.01) }
_ = try? await transcribe(noise)
_ = try? await transcribe(noise)
emit(["ready": true, "load_s": loadS, "warm_s": Date().timeIntervalSince(t0)])

while true {
    guard let head = readExactly(4) else { break }
    let n = Int(head.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian })
    if n == 0 { break }
    guard let body = readExactly(n * 4) else { break }
    let samples: [Float] = body.withUnsafeBytes { raw in
        (0..<n).map { Float(bitPattern: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self).littleEndian) }
    }
    let start = Date()
    do {
        let r = try await transcribe(samples)
        emit(["text": r.text, "seconds": Date().timeIntervalSince(start), "audio_s": Double(n) / 16000,
              "confidence": Double(r.confidence)])
    } catch {
        emit(["error": "\(error)"])
    }
}
await asr.cleanup()
