"""recall.py — the app's memory of past screens, for recall questions.

Frames the app reads (screen.ScreenWatcher) -> OCR (Apple Vision) -> pages and visits -> the context text.

  page key   the eval's rules, unchanged (moved here from eval/real/retrieval.py so the app and the eval share them):
             browsers by tab title, other apps by window; text = OCR lines inside the front window, below a
             browser's address bar. A frame whose front window changed across the grab is skipped.
  visits     a run of frames on one page; it ends when a frame from another page arrives. Each frame lasts
             until the next one.
  store      a visit's screens enter the page store when the visit ends, so the store holds every page as of its
             last finished visit. The page on screen now is in the context as the screen read, not from here.
             Every page has a version that goes up when a visit of it ends: its text and its time header change
             then, and so must its stored KV piece.
  context    the text layout of eval/recall/rag_arm.py, unchanged (moved here, rag_arm imports it): the timeline
             (one line per finished visit), then the pages slot: retrieved whole pages under a token cap, best
             BM25 first until the cap, shown oldest first, each under a time header.

No model here. OCR runs on its own thread (Ocr) and pauses during a turn, so it never competes with speech
recognition for the Neural Engine.
"""
from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np

from memory import Memory, Page

# ---------------------------------------------------------------- page key and text

BROWSERS = {"Google Chrome", "Safari", "Arc", "Firefox", "Microsoft Edge", "Brave Browser", "Chromium", "Opera",
            "Vivaldi", "Orion", "Zen"}
TOOLBAR_BAND = 130     # pt from a browser window's top within which its address bar is looked for
_DOMAIN = re.compile(r"^[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?:[/:?#]|$)")


def clean_title(t: str) -> str:
    """A document's name without status: symbol characters (emoji, audio badges, spinners) and a leading
    unread counter like "(3) " are dropped."""
    t = "".join(c for c in (t or "") if unicodedata.category(c) != "So")
    t = re.sub(r"^\(\d+\+?\)\s*", "", t.strip())
    return re.sub(r"\s+", " ", t).strip()


def domain_start(line: str):
    """The domain a line starts with, after leading icons (short tokens, punctuation), or None."""
    for tok in line.split():
        tok = tok.lstrip("(<[•·º°").lower()
        if len(tok) <= 2 or not any(c.isalnum() for c in tok):
            continue
        m = _DOMAIN.match(tok)
        return m.group(0).rstrip("/:?#") if m else None
    return None


def address_bar(win, lines, boxes):
    """(bottom y, domain) of the address bar line in a browser window, or None."""
    best = None
    for line, (x, y, w, h) in zip(lines, boxes):
        cy = y + h / 2
        if win["y"] <= cy <= win["y"] + TOOLBAR_BAND and win["x"] <= x + w / 2 <= win["x"] + win["w"]:
            d = domain_start(line)
            if d and (best is None or cy > best[2]):
                best = (y + h, d, cy)
    return best[:2] if best else None


def page_key(win, bar, windows) -> str:
    if win["app"] in BROWSERS:
        return clean_title(win.get("title")) or (bar[1] if bar else win["app"])
    wid = win.get("id")
    if wid is not None:
        return f'{win["app"]} window {wid}'
    # Older recordings saved no window number. Windows move, so bounds are a poor stand-in (a moved
    # terminal became two pages); the app is the stand-in when it has one window on screen.
    if sum(w["app"] == win["app"] for w in windows) == 1:
        return f'{win["app"]} window'
    return f'{win["app"]} window at {win["x"]},{win["y"]} {win["w"]}x{win["h"]}'


def page_lines(win, lines, boxes, bar, with_boxes: bool = False):
    top = bar[0] if (win["app"] in BROWSERS and bar) else win["y"]
    out, kept = [], []
    for line, (x, y, w, h) in zip(lines, boxes):
        cx, cy = x + w / 2, y + h / 2
        if win["x"] <= cx <= win["x"] + win["w"] and top < cy <= win["y"] + win["h"]:
            out.append(line)
            kept.append([x, y, w, h])
    return (out, kept) if with_boxes else out


ROW_SEP = " | "


def ocr_rows(lines, boxes) -> list[str]:
    """OCR lines on one visual row joined left to right: two lines share a row when their
    vertical centres differ by less than half the smaller line's height. Rows top to bottom. Vision returns each
    table cell as its own line, so without this a value loses the label on its row."""
    items = sorted(zip(lines, boxes), key=lambda lb: lb[1][1] + lb[1][3] / 2)
    rows: list[list] = []
    for line, (x, y, w, h) in items:
        cy = y + h / 2
        r = rows[-1] if rows else None
        if r is not None and abs(cy - r[0]) < 0.5 * min(h, r[1]):
            r[2].append((x, line))
        else:
            rows.append([cy, h, [(x, line)]])
    return [ROW_SEP.join(t for _, t in sorted(r[2], key=lambda xt: xt[0])) for r in rows]


def changed_across_grab(r) -> bool:
    before = r.get("windows_before")
    if before is None or not r["windows"]:
        return False
    a, b = before[0] if before else {}, r["windows"][0]
    return (a.get("id"), a.get("app"), a.get("title")) != (b.get("id"), b.get("app"), b.get("title"))


def windows():
    """Visible app windows (layer 0), front to back: app, title, bounds in points."""
    import Quartz
    out = []
    infos = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID)
    for w in infos or []:
        if w.get("kCGWindowLayer") != 0:
            continue
        b = w.get("kCGWindowBounds") or {}
        if b.get("Width", 0) < 100 or b.get("Height", 0) < 100:
            continue
        out.append({"id": int(w.get("kCGWindowNumber", -1)), "app": str(w.get("kCGWindowOwnerName") or ""),
                    "title": str(w.get("kCGWindowName") or ""),
                    "x": int(b.get("X", 0)), "y": int(b.get("Y", 0)), "w": int(b.get("Width", 0)),
                    "h": int(b.get("Height", 0))})
    return out


# ---------------------------------------------------------------- context text (rag_arm.py's layout)

TIMELINE_HEAD = "(Timeline of the last hour from the user's screen, one line per page visit, oldest first.)\n"
PAGES_HEAD = ("(Pages seen earlier in the last hour that may be relevant. None of them are on screen any more; "
              "use them for questions about what was on screen before.)\n")
HINT = "If the answer is not in these pages or on the screen, say you didn't see it.\n"


def hm(dt: datetime) -> str:
    return dt.strftime("%I:%M").lstrip("0")


def span(a: datetime, b: datetime) -> str:
    """8:33–8:35 AM (AM/PM once when both ends share it)."""
    pa, pb = a.strftime("%p"), b.strftime("%p")
    if hm(a) == hm(b) and pa == pb:
        return f"{hm(a)} {pa}"
    return f"{hm(a)}–{hm(b)} {pb}" if pa == pb else f"{hm(a)} {pa}–{hm(b)} {pb}"


def dur(s: float) -> str:
    return f"{round(s)} s" if s < 60 else f"{round(s / 60)} min"


def timeline_text(visits, at: Callable[[float], datetime], title: Callable[[str], str]) -> str:
    return TIMELINE_HEAD + "\n".join(f"{span(at(t0), at(t1))} ({dur(t1 - t0)}) {title(k)}" for k, t0, t1 in visits)


def page_head(p, at: Callable[[float], datetime], n_visits: int) -> str:
    n = n_visits
    return (f"[{span(at(p.first_s), at(p.last_s))}, {dur(p.dwell_s)} on screen, "
            f"{n} visit{'s' if n != 1 else ''}] ")


EXCERPT_HEAD = "(Parts of the timeline near the pages below and near times in the question, oldest first.)\n"
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\s*([ap]\.?m\.?)?|\b(\d{1,2})\s*([ap]\.?m\.?)(?![a-z])", re.I)


def clock_times(question: str, start: datetime, end_s: float) -> list[float]:
    """Clock times named in a question, as seconds after `start`. A time without AM/PM takes the reading closest to
    the span [start, start + end_s]."""
    out = []
    for m in _CLOCK.finditer(question):
        hh, mm, ap = (int(m[1]), int(m[2]), m[3]) if m[1] else (int(m[4]), 0, m[5])
        if hh > 23 or mm > 59:
            continue
        cands = []
        for h24 in ({(hh % 12) + (12 if ap.lower().startswith("p") else 0)} if ap else {hh % 12, hh % 12 + 12}):
            t0 = (start.replace(hour=h24, minute=mm, second=0, microsecond=0) - start).total_seconds()
            for t in (t0 - 86400, t0, t0 + 86400):     # the day before or after too: an hour can cross midnight
                cands.append((0 if 0 <= t <= end_s else min(abs(t), abs(t - end_s)), t))
        out.append(min(cands)[1])
    return out


def timeline_excerpt(m: Memory, visits, question: str, at: Callable[[float], datetime], title: Callable[[str], str],
                     start: datetime, top: int = 3, around: int = 2, minutes: float = 2.0) -> str:
    """Retrieve the timeline: the visits of the `top` best BM25 pages with `around` visits
    on each side, and the visits within `minutes` of each clock time in the question; in the timeline's format,
    deduplicated, oldest first. Empty if nothing matched."""
    keys = {p.url for p, _ in m.rank_pages(question)[:top]}
    pick = set()
    for i, (k, _, _) in enumerate(visits):
        if k in keys:
            pick.update(range(max(0, i - around), min(len(visits), i + around + 1)))
    end_s = visits[-1][2] if visits else 0.0
    for t in clock_times(question, start, end_s):
        pick.update(i for i, (_, t0, t1) in enumerate(visits) if t0 <= t + 60 * minutes and t1 >= t - 60 * minutes)
    if not pick:
        return ""
    lines = [f"{span(at(t0), at(t1))} ({dur(t1 - t0)}) {title(k)}" for i, (k, t0, t1) in enumerate(visits) if i in pick]
    return EXCERPT_HEAD + "\n".join(lines)


def pick_whole_pages(m: Memory, question: str, cap: int, used: int, count, head_of, max_pages=None):
    """Whole pages, best BM25 first, while they fit under `cap` tokens (`used` already spent); the page that
    doesn't fit is cut at a line and ends the list. Returns ([(page, lines kept)], tokens used)."""
    picked = []
    for p, _ in m.rank_pages(question):
        cost = count(head_of(p) + p.text([]))
        if used + cost > cap:
            break
        used += cost
        kept = []
        for b in p.blocks:
            c = count(b + "\n")
            if used + c > cap:
                break
            kept.append(b)
            used += c
        if not kept:
            used -= cost
            break
        picked.append((p, kept))
        if len(kept) < len(p.blocks) or len(picked) == max_pages:
            break
    return picked, used


# ---------------------------------------------------------------- the live store

@dataclass
class _Visit:
    key: str
    t0: float
    screens: list = field(default_factory=list)       # (t, title, text)


class Store:
    """Pages and visits from the frames the app reads. Times are seconds since `start` (wall clock).
    Thread-safe: frames come from the OCR thread, questions from the GPU thread."""

    def __init__(self, start: Optional[datetime] = None):
        self.start = start or datetime.now()
        self.m = Memory()
        self.visits: list[tuple[str, float, float]] = []   # finished visits (key, t0, t1)
        self.n_visits: dict[str, int] = {}
        self.version: dict[str, int] = {}                  # key -> number of finished visits
        self.cur: Optional[_Visit] = None
        self.views: dict[str, Page] = {}                   # "view:<fid>" -> a kept screen's view text
        self.lock = threading.RLock()

    def at(self, t: float) -> datetime:
        return self.start + timedelta(seconds=t)

    def add(self, t: float, key: str, title: str, text: str) -> Optional[str]:
        """One screen of a page at time t. Returns the key of the visit this frame ended, if any."""
        with self.lock:
            ended = None
            if self.cur is not None and self.cur.key != key:
                ended = self._end(t)
            if self.cur is None:
                self.cur = _Visit(key, t)
            self.cur.screens.append((t, title, text))
            return ended

    @staticmethod
    def frame_info(wins: list, lines: list, boxes: list) -> tuple[str, str, list]:
        """(page key, title, page-area lines) of a frame, by the page key rules at the top."""
        win = wins[0]
        bar = address_bar(win, lines, boxes) if win["app"] in BROWSERS else None
        key = page_key(win, bar, wins)
        return key, clean_title(win.get("title")) or key, page_lines(win, lines, boxes, bar)

    def add_frame(self, t: float, wins: list, lines: list, boxes: list) -> Optional[str]:
        """A frame with its window list (front to back) and OCR lines with boxes in screen points."""
        if not wins:
            return None
        key, title, text = self.frame_info(wins, lines, boxes)
        return self.add(t, key, title, "\n".join(text))

    def _end(self, t_end: float) -> str:
        v = self.cur
        for i, (t0, title, text) in enumerate(v.screens):
            t1 = v.screens[i + 1][0] if i + 1 < len(v.screens) else t_end
            self.m.add_screen(t0, t1, v.key, title, text)
        self.visits.append((v.key, v.t0, t_end))
        self.n_visits[v.key] = self.n_visits.get(v.key, 0) + 1
        self.version[v.key] = self.version.get(v.key, 0) + 1
        self.cur = None
        return v.key

    def current_key(self) -> Optional[str]:
        return self.cur.key if self.cur is not None else None

    # -- context text
    def timeline(self) -> str:
        with self.lock:
            return timeline_text(self.visits, self.at, lambda k: self.m.pages[k].title)

    def page(self, key: str) -> Optional[Page]:
        """A page of the store, or a kept screen's view ("view:<fid>")."""
        with self.lock:
            return self.views.get(key) or self.m.pages.get(key)

    def head(self, key: str) -> str:
        with self.lock:
            return page_head(self.page(key), self.at, self.n_visits.get(key, 1))

    def page_text(self, key: str, n_lines: Optional[int] = None) -> str:
        """A page's slot text as it goes into the context: "\\n\\n" + time header + page (first n_lines lines)."""
        with self.lock:
            p = self.page(key)
            return "\n\n" + self.head(key) + p.text(p.blocks if n_lines is None else p.blocks[:n_lines])

    def set_view(self, fid: int, screen) -> str:
        """A kept screen's view text as a memory slot (Andrew, 2026-11-12): the page-area lines of the kept frame
        and the frames it stands in for, headed like its page (time, title, address). Its version bumps when the text
        grows (dropped frames keep arriving). Returns its key "view:<fid>"."""
        vk = f"view:{fid}"
        with self.lock:
            lines = list(screen.lines)
            old = self.views.get(vk)
            if old is None or old.blocks != lines:
                orig = self.m.pages.get(screen.key)
                self.views[vk] = Page(url=screen.key, title=screen.title, blocks=lines, first_s=screen.t,
                                      last_s=screen.t, dwell_s=orig.dwell_s if orig else 0.0, screens=1)
                self.version[vk] = self.version.get(vk, 0) + 1
                self.n_visits[vk] = self.n_visits.get(screen.key, 1)
            return vk

    def view_slot(self, vk: str, budget: int, count) -> tuple:
        """(key, version, lines shown): the view's lines from the top, as many as fit in `budget` tokens (at least one)."""
        with self.lock:
            p, n = self.views[vk], 0
            while n < len(p.blocks) and count(self.page_text(vk, n + 1)) <= budget:
                n += 1
            return (vk, self.version[vk], max(n, 1))

    def drop_view(self, fid: int):
        with self.lock:
            self.views.pop(f"view:{fid}", None)

    def excerpt(self, question: str) -> str:
        with self.lock:
            return timeline_excerpt(self.m, self.visits, question, self.at, lambda k: self.m.pages[k].title, self.start)

    def pick(self, question: str, cap: int, count, used: int = 0) -> list[tuple]:
        """The pages slot for a question: [(key, version, lines shown)] oldest first. `used`: tokens of the cap
        already spent (timeline, headers)."""
        with self.lock:
            picked, _ = pick_whole_pages(self.m, question, cap, used, count, lambda p: self.head(p.url))
            shown = sorted(picked, key=lambda x: x[0].first_s)
            return [(p.url, self.version[p.url], len(kept)) for p, kept in shown]


# ---------------------------------------------------------------- which screens to keep

CALL_APPS = re.compile(r"\b(?:meet|zoom|facetime|teams|webex)\b", re.I)


class FrameKeeper:
    """Keep rule K: a frame becomes memory if, against the frames already kept for its page,
    it shows a page-area OCR line not seen before, or it differs from the page's last kept frame by the change gate's
    own rule (screen.py: at least min(CELL_MIN, CELL_FRAC of the screen) changed cells, status area ignored).
    Video-call windows are never kept."""

    def __init__(self):
        self.seen: dict[str, set] = {}
        self.last_small: dict[str, object] = {}

    @staticmethod
    def is_call(win) -> bool:
        return bool(CALL_APPS.search(win.get("app", "") + " " + (win.get("title") or "")))

    def decide(self, key: str, lines: list[str], small) -> Optional[str]:
        """Returns the reason the frame is kept ("new text" / "changed"), or None. Records it as kept if so."""
        import screen
        seen = self.seen.setdefault(key, set())
        new = [ln for ln in (x.strip() for x in lines) if ln and ln not in seen]
        reason = "new text" if new or key not in self.last_small else None
        if reason is None:
            prev = _unpack(self.last_small[key])
            if small is not None and prev is not None:
                if prev.shape != small.shape:
                    reason = "changed"
                else:
                    ch = screen.cell_changes(small, prev)
                    m = screen.cell_mask(*ch.shape)
                    n = int(m.sum())
                    if int((ch & m).sum()) >= min(screen.CELL_MIN, max(1, int(screen.CELL_FRAC * n))):
                        reason = "changed"
        if reason is not None:
            seen.update(x.strip() for x in lines if x.strip())
            self.last_small[key] = _pack(small)
        return reason


def _pack(a):
    """A page's last kept thumbnail, zlib-compressed (lossless): ~1.1 MB each as an array, one per page, grew with
    the hour (M3, 2026-11-13)."""
    import zlib
    if a is None:
        return None
    a = np.asarray(a)
    return (a.shape, a.dtype.str, zlib.compress(np.ascontiguousarray(a).tobytes(), 1))


def _unpack(p):
    import zlib
    if p is None:
        return None
    shape, dt, z = p
    return np.frombuffer(zlib.decompress(z), dtype=np.dtype(dt)).reshape(shape)


# ---------------------------------------------------------------- OCR (Apple Vision)

def ocr_image(img) -> tuple[list[str], list[list[float]]]:
    """Accurate-mode OCR of a PIL screenshot (device pixels, 2 per point). Returns (lines, boxes in points:
    x, y from the top-left, w, h), as eval/real/ocr.py writes them."""
    import objc
    import Quartz
    import Vision
    from Foundation import NSData
    rgb = img.convert("RGBA")
    W, H = rgb.size
    # The provider owns a copy of the pixels (CFData) and every Objective-C object made here is released when the pool
    # drains. CGDataProviderCreateWithData on Python bytes, or no pool on this long-lived thread, leaked one full
    # RGBA frame (~24 MB at 3024x1964) per call (M3 baseline, 2026-11-13).
    with objc.autorelease_pool():
        data = NSData.dataWithBytes_length_(rgb.tobytes(), W * H * 4)
        provider = Quartz.CGDataProviderCreateWithCFData(data)
        cg = Quartz.CGImageCreate(W, H, 8, 32, W * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
                                  Quartz.kCGImageAlphaNoneSkipLast, provider, None, False,
                                  Quartz.kCGRenderingIntentDefault)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(False)
        Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None).performRequests_error_([req], None)
        res = req.results() or []
        Wp, Hp = W / 2, H / 2
        lines = [str(o.topCandidates_(1)[0].string()) for o in res]
        boxes = [[round(b.origin.x * Wp, 1), round((1 - b.origin.y - b.size.height) * Hp, 1),
                  round(b.size.width * Wp, 1), round(b.size.height * Hp, 1)] for b in (o.boundingBox() for o in res)]
    return lines, boxes


class Ocr(threading.Thread):
    """Reads frames into a Store, oldest first, while `quiet()` holds (no turn active). Frames are dropped
    after reading; nothing is written to disk. `on_visit_end(key)` is called when a frame ends a visit.
    `on_keep(fid, image, meta)`, if given, is called for frames keep rule K keeps as screen memory; video-call
    windows are never kept."""

    def __init__(self, store: Store, quiet: Callable[[], bool], on_visit_end: Callable[[str], None],
                 ocr: Callable = ocr_image, t0: Optional[float] = None, on_keep: Optional[Callable] = None,
                 on_unit_text: Optional[Callable] = None):
        super().__init__(daemon=True, name="ocr")
        self.store, self.quiet, self.on_visit_end, self.ocr = store, quiet, on_visit_end, ocr
        self.on_keep, self.keeper = on_keep, FrameKeeper()
        # Units: a dropped frame's text goes to the kept screen standing in for it (the page's last kept frame;
        # before the page's first keep, the next one). on_unit_text(fid, lines).
        self.on_unit_text = on_unit_text
        self._last_kept: dict[str, int] = {}
        self._waiting: dict[str, list] = {}
        self.cv = threading.Condition()
        self.q: list[tuple] = []
        self.t0 = t0
        self._stop = False
        self.stats = {"frames": 0, "skipped_mid_switch": 0, "ms": [], "kept": 0}

    def submit(self, t_wall: float, img, wins_before: Optional[list], wins_after: list, fid: Optional[int] = None):
        with self.cv:
            self.q.append((t_wall, img, wins_before, wins_after, fid))
            self.cv.notify_all()

    def poke(self):
        with self.cv:
            self.cv.notify_all()

    def stop(self):
        with self.cv:
            self._stop = True
            self.cv.notify_all()

    def run(self):
        import time
        while True:
            with self.cv:
                while not self._stop and not (self.q and self.quiet()):
                    self.cv.wait(0.1)
                if self._stop:
                    return
                t_wall, img, before, after, fid = self.q.pop(0)
            if changed_across_grab({"windows": after, "windows_before": before}):
                self.stats["skipped_mid_switch"] += 1
                continue
            t = time.perf_counter()
            lines, boxes = self.ocr(img)
            self.stats["ms"] = (self.stats["ms"] + [round((time.perf_counter() - t) * 1000, 1)])[-200:]
            self.stats["frames"] += 1
            t_rel = t_wall - (self.t0 if self.t0 is not None else self.store.start.timestamp())
            ended = self.store.add_frame(t_rel, after, lines, boxes)
            if ended is not None:
                self.on_visit_end(ended)
            if self.on_keep is not None and fid is not None and after and not FrameKeeper.is_call(after[0]):
                import screen
                key, title, text = Store.frame_info(after, lines, boxes)
                if self.keeper.decide(key, text, screen.shrink(img)):
                    self.stats["kept"] += 1
                    before = self._waiting.pop(key, [])
                    self._last_kept[key] = fid
                    self.on_keep(fid, img, {"t": t_rel, "key": key, "title": title,
                                            "lines": list(dict.fromkeys(before + list(text)))})
                elif self.on_unit_text is not None:
                    if key in self._last_kept:
                        self.on_unit_text(self._last_kept[key], list(text))
                    else:
                        w = self._waiting.setdefault(key, [])
                        w.extend(ln for ln in text if ln not in w)
                        del w[:-2000]                           # bounded: a page that is never kept
