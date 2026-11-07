// winlist: on-screen app windows, front to back, as JSON lines. No window titles (privacy).
import CoreGraphics
import Foundation
let opts: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
let list = (CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]]) ?? []
var out: [[String: Any]] = []
for w in list {
  guard (w[kCGWindowLayer as String] as? Int ?? -1) == 0 else { continue }
  let b = w[kCGWindowBounds as String] as? [String: Double] ?? [:]
  out.append(["id": w[kCGWindowNumber as String] as? Int ?? 0,
              "app": w[kCGWindowOwnerName as String] as? String ?? "?",
              "x": b["X"] ?? 0, "y": b["Y"] ?? 0, "w": b["Width"] ?? 0, "h": b["Height"] ?? 0])
}
let data = try! JSONSerialization.data(withJSONObject: out)
print(String(data: data, encoding: .utf8)!)
