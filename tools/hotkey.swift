// hotkey: push-to-talk from any app. Prints "down" / "up" when the chosen modifier key is pressed /
// released. Listens to modifier-key changes only (flagsChanged), never to typed characters.
// Needs macOS Input Monitoring permission for the app that runs it (e.g. Terminal).
//   tools/hotkey [right_option|right_command|right_control|fn]
import CoreGraphics
import Foundation

let keys: [String: (Int64, CGEventFlags)] = [
  "right_option": (61, .maskAlternate), "right_command": (54, .maskCommand),
  "right_control": (62, .maskControl), "fn": (63, .maskSecondaryFn),
]
let name = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "right_option"
guard let chosen = keys[name] else { print("error: unknown key \(name)"); exit(2) }
var gKeycode: Int64 = chosen.0          // globals: a C callback can't capture locals
var gFlag: CGEventFlags = chosen.1

if !CGPreflightListenEventAccess() {
  _ = CGRequestListenEventAccess()       // shows the system prompt once
  print("error: no Input Monitoring permission (System Settings > Privacy & Security > Input Monitoring)")
  fflush(stdout)
  exit(1)
}

var isDown = false
let mask = CGEventMask(1 << CGEventType.flagsChanged.rawValue)
let callback: CGEventTapCallBack = { _, type, event, _ in
  if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput { return Unmanaged.passUnretained(event) }
  if event.getIntegerValueField(.keyboardEventKeycode) == gKeycode {
    let down = event.flags.contains(gFlag)
    if down != isDown {
      isDown = down
      print(down ? "down" : "up")
      fflush(stdout)
    }
  }
  return Unmanaged.passUnretained(event)
}
guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .listenOnly,
                                  eventsOfInterest: mask, callback: callback, userInfo: nil) else {
  print("error: could not create the event tap (Input Monitoring permission?)")
  fflush(stdout)
  exit(1)
}
let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)
CGEvent.tapEnable(tap: tap, enable: true)
print("ready \(name)")
fflush(stdout)
CFRunLoopRun()
