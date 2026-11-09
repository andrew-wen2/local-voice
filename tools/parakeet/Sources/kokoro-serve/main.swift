// kokoro-serve: Kokoro-82M text-to-speech through FluidAudio's KokoroAne (Core ML, Neural Engine + CPU),
// the candidate replacement for MLX Kokoro on the GPU (ARCHITECTURE.md §3.7). English, voice af_heart
// (the app's voice). English text-to-phonemes runs in Swift (misaki lexicon + a small G2P model), so the
// Python spaCy/misaki stack isn't needed.
//
//   kokoro-serve --models DIR --download          fetch the Core ML bundles into DIR once (needs network)
//   kokoro-serve --models DIR --bench OUTDIR [--noise gpu|cpu]
//        synthesize a fixed set of answer-like sentences; per sentence: wall time, audio length, stage
//        timings; process memory (phys_footprint) after load and at the end; WAVs into OUTDIR to listen to.
//   --pipeline legacy|v3: FluidAudio's original seven-stage chain (default) or its hybrid ANE/GPU chain.
//   kokoro-serve --models DIR [--noise gpu|cpu]   serve: one UTF-8 line of text in, then
//        "@@" + JSON {"n", "seconds", "audio_s"} and n Float32 little-endian samples (24 kHz) out.
// Lines without the "@@" prefix are library logging.

import CoreML
import Darwin
import FluidAudio
import Foundation

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}

func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : -1
}

func emit(_ obj: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
    FileHandle.standardOutput.write("@@".data(using: .utf8)! + data + "\n".data(using: .utf8)!)
}

guard let modelsPath = arg("--models") else {
    FileHandle.standardError.write("usage: kokoro-serve --models DIR [--download | --bench OUTDIR] [--noise gpu|cpu]\n".data(using: .utf8)!)
    exit(2)
}
let modelsDir = URL(fileURLWithPath: modelsPath, isDirectory: true)
var units = KokoroAneComputeUnits.default
// Defaults chosen by measurement (ARCHITECTURE.md §3.7): the all-fp32 noise stage on the CPU (on the GPU it
// cost ~1 GB more memory), and the alignment stage on the GPU: its CPU fallback (BNNS) traps on some exact
// input lengths ("The observatory was 36 meters high." crashed 1 in 58 phrases; FluidAudio 0.18.2, both the
// legacy and v3 chains). On the GPU: 0 crashes in 1,353 phrases, +10 MB.
units.noise = .cpuOnly
units.alignment = .cpuAndGPU
if arg("--noise") == "gpu" { units.noise = .cpuAndGPU }
// --stage NAME=ane|cpu|gpu (repeatable): place one Kokoro stage (albert, postAlbert, alignment, prosody, noise,
// vocoder, tail) on specific compute units.
for (i, a) in CommandLine.arguments.enumerated() where a == "--stage" && i + 1 < CommandLine.arguments.count {
    let kv = CommandLine.arguments[i + 1].split(separator: "=").map(String.init)
    guard kv.count == 2 else { continue }
    let u: MLComputeUnits = kv[1] == "gpu" ? .cpuAndGPU : kv[1] == "cpu" ? .cpuOnly : .cpuAndNeuralEngine
    switch kv[0] {
    case "albert": units.albert = u
    case "postAlbert": units.postAlbert = u
    case "alignment": units.alignment = u
    case "prosody": units.prosody = u
    case "noise": units.noise = u
    case "vocoder": units.vocoder = u
    case "tail": units.tail = u
    default: break
    }
}
let version: KokoroAneVersion = arg("--pipeline") == "v3" ? .v3 : .legacy

let t0 = Date()
let fp0 = footprintMB()
let tts = KokoroAneManager(variant: .english, defaultVoice: "af_heart", directory: modelsDir, computeUnits: units,
                           version: version)
do {
    try await tts.initialize()
} catch {
    FileHandle.standardError.write("init failed: \(error)\n".data(using: .utf8)!)
    exit(1)
}
let loadS = Date().timeIntervalSince(t0)
if CommandLine.arguments.contains("--download") {
    print("models ready in \(modelsDir.path) (\(String(format: "%.1f", loadS)) s)")
    exit(0)
}
// warm-up: the first call compiles / specializes the Core ML models
let tw = Date()
_ = try await tts.synthesizeDetailed(text: "Okay.")
let warmS = Date().timeIntervalSince(tw)
let fpLoaded = footprintMB()

if let outPath = arg("--bench") {
    let out = URL(fileURLWithPath: outPath, isDirectory: true)
    try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let sentences = [
        "Okay.",
        "The rent was $3,488 per month.",
        "Your return flight MA 594 from Lisbon departs at 9:30 PM on December 16th.",
        "The city council voted 6-2 to approve the $2.8 million for the new skate park.",
        "Marcus Lindqvist sent the email with the subject 'Updated offer timeline'. It was sent at 2:12 PM.",
        "I didn't see that on your screen in the last hour.",
        "The serial number is K7QX4MV29T, and it's listed under General settings, in the About section, near the bottom of the page.",
    ]
    var rows: [[String: Any]] = []
    var peak = fpLoaded
    for (i, s) in sentences.enumerated() {
        let t = Date()
        let r = try await tts.synthesizeDetailed(text: s)
        let wall = Date().timeIntervalSince(t)
        peak = max(peak, footprintMB())
        let wav = try AudioWAV.data(from: r.samples, sampleRate: Double(r.sampleRate))
        try wav.write(to: out.appendingPathComponent(String(format: "s%02d.wav", i)))
        rows.append(["i": i, "chars": s.count, "wall_ms": Int(wall * 1000), "audio_s": (r.durationSeconds * 100).rounded() / 100,
                     "rtf": (wall / max(r.durationSeconds, 1e-6) * 1000).rounded() / 1000, "stages_ms": Int(r.timings.totalMs)])
        print(String(format: "s%02d %4d chars  %5d ms for %.2f s audio  (x%.2f real time)  %@", i, s.count, Int(wall * 1000),
                     r.durationSeconds, wall / max(r.durationSeconds, 1e-6), s.prefix(60) as CVarArg))
    }
    emit(["load_s": loadS, "warm_s": warmS, "footprint_mb_start": fp0, "footprint_mb_loaded": fpLoaded,
          "footprint_mb_peak_sampled": peak, "footprint_mb_end": footprintMB(), "noise": arg("--noise") ?? "cpu", "rows": rows])
    exit(0)
}

emit(["ready": true, "load_s": loadS, "warm_s": warmS, "footprint_mb": fpLoaded])
while let line = readLine(strippingNewline: true) {
    if line.isEmpty { continue }
    let t = Date()
    do {
        let r = try await tts.synthesizeDetailed(text: line)
        emit(["n": r.samples.count, "seconds": Date().timeIntervalSince(t), "audio_s": r.durationSeconds])
        r.samples.withUnsafeBufferPointer { FileHandle.standardOutput.write(Data(buffer: $0)) }
    } catch {
        emit(["n": 0, "error": "\(error)"])
    }
}
