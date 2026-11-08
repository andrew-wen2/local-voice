"""Screen memory for recall at question time (ARCHITECTURE.md §9.9).

The hour is kept outside the model and searched per question; only a slice goes into the context.

  store    one entry per page (URL): its title, the text blocks seen on it (lines, deduplicated across
           screens, so a scrolled page grows), when it was first and last on screen, and total dwell.
  index    an exact-word index (BM25) over blocks and over whole pages.
  retrieve rank pages (or blocks) for a question and fill a token budget.
  check    "does the thing the question names appear anywhere in memory?" If a content word of the
           question was never on screen, answer "I didn't see that" instead of guessing.

No model here: word matching only. Embeddings are a later arm.
"""
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# General English function words. Kept generic on purpose: the check must not be tuned to the benchmark's
# phrasing (see eval/recall/retrieval.py for the dev / held-out split).
STOP = set("""
a an the and or but if then so of to in on at by for with from about into over under after before up down
out off as is are was were be been being am do does did done have has had having it its this that these those
there here what which who whom whose when where why how much many long tall far old big
i me my mine we us our you your he him his she her they them their
can could would should will shall may might must not no yes any some all each every both either neither
than too very just also only again ever still even
say said says tell told show shown see saw seen look looking looked
""".split())

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def stem(w: str) -> str:
    """Crude suffix stripping, applied the same way to memory and questions."""
    w = re.sub(r"'s$", "", w)
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def terms(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower())]


def content_terms(question: str) -> list[str]:
    out = []
    for w in _WORD.findall(question.lower()):
        w = re.sub(r"'s$", "", w)
        if w not in STOP:
            out.append(stem(w))
    return out


_RAW = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]*(?:'s)?|[,;:?!()]")


def named_phrases(question: str) -> list[str]:
    """Maximal runs of capitalized words or words with digits, ignoring the question's first word and "I".
    "What was the rent for 974 Halsey Court, Apt 3B?" -> ["974 Halsey Court", "Apt 3B"]."""
    out, run = [], []
    toks = _RAW.findall(question)
    for i, t in enumerate(toks):
        w = re.sub(r"'s$", "", t).rstrip(".")
        named = bool(w) and w != "I" and i > 0 and (w[0].isupper() or any(c.isdigit() for c in w))
        if named:
            run.append(w)
        else:
            if run:
                out.append(" ".join(run))
            run = []
    if run:
        out.append(" ".join(run))
    return out


@dataclass
class Page:
    url: str
    title: str
    blocks: list[str] = field(default_factory=list)
    first_s: float = 0.0
    last_s: float = 0.0
    dwell_s: float = 0.0
    screens: int = 0

    def text(self, blocks=None) -> str:
        return f'Page "{self.title}" ({self.url}):\n' + "\n".join(self.blocks if blocks is None else blocks)


class BM25:
    def __init__(self, docs: list[list[str]], k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, len(docs))
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


class Memory:
    def __init__(self):
        self.pages: dict[str, Page] = {}
        self._index = None

    # -- capture
    def add_screen(self, t0: float, t1: float, url: str, title: str, text: str):
        """One screen: the active tab's URL and title (from the window, not OCR) and its visible text."""
        p = self.pages.get(url)
        if p is None:
            p = self.pages[url] = Page(url, title, first_s=t0)
        seen = set(p.blocks)
        for line in text.split("\n"):
            line = line.strip()
            if line and line not in seen:
                p.blocks.append(line)
                seen.add(line)
        p.title = title
        p.last_s = max(p.last_s, t1)
        p.dwell_s += t1 - t0
        p.screens += 1
        self._index = None

    # -- index
    def index(self):
        if self._index is None:
            pages = list(self.pages.values())
            blocks = [(i, b) for i, p in enumerate(pages) for b in p.blocks]
            page_docs = [terms(p.title + " " + p.url) * 2 + terms(" ".join(p.blocks)) for p in pages]
            block_docs = [terms(pages[i].title) + terms(b) for i, b in blocks]
            vocab = set(t for d in page_docs for t in d)
            self._index = dict(pages=pages, blocks=blocks, page_bm25=BM25(page_docs),
                               block_bm25=BM25(block_docs), vocab=vocab)
        return self._index

    # -- check
    def unseen(self, question: str) -> list[str]:
        """Content words of the question that never appeared on screen (title, URL or text)."""
        vocab = self.index()["vocab"]
        return [t for t in content_terms(question) if t not in vocab]

    def unseen_named(self, question: str) -> list[str]:
        """Named phrases of the question (runs of capitalized words, numbers and codes) that never appeared
        on screen as a phrase. Lowercase words are not checked: they are mostly what's asked about
        ("vote", "start", "meeting"), not the thing named."""
        ix = self.index()
        if "phrases" not in ix:
            ix["phrases"] = [" " + " ".join(terms(p.title + " . " + p.url + " . " + " ".join(p.blocks))) + " "
                             for p in ix["pages"]]
        out = []
        for ph in named_phrases(question):
            key = " " + " ".join(terms(ph)) + " "
            if key.strip() and not any(key in doc for doc in ix["phrases"]):
                out.append(ph)
        return out

    # -- retrieve
    def rank_pages(self, question: str, now_s: float | None = None, recency: float = 0.0) -> list[tuple[Page, float]]:
        """Pages by BM25 over the whole page (title and URL counted twice), plus an optional recency prior:
        score * (1 + recency * exp(-age / 30 min))."""
        ix = self.index()
        sc = ix["page_bm25"].scores(terms(question))
        if recency and now_s is not None:
            sc = [s * (1 + recency * math.exp(-(now_s - p.last_s) / 1800)) for s, p in zip(sc, ix["pages"])]
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        return [(ix["pages"][i], sc[i]) for i in order if sc[i] > 0]

    def rank_blocks(self, question: str) -> list[tuple[Page, str, float]]:
        ix = self.index()
        sc = ix["block_bm25"].scores(terms(question))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        return [(ix["pages"][ix["blocks"][i][0]], ix["blocks"][i][1], sc[i]) for i in order if sc[i] > 0]

    def retrieve(self, question: str, budget: int, count, mode: str = "pages", **kw) -> list[tuple[Page, list[str]]]:
        """Fill `budget` tokens (count: str -> int). Returns [(page, blocks included)] in rank order.

        pages   whole pages in rank order, one result per page; the page that doesn't fit is cut at a block.
        chunks  plain RAG: top blocks by BM25 regardless of page, grouped under their page for display.
        """
        out: dict[str, tuple[Page, list[str]]] = {}
        used = 0
        if mode == "pages":
            for p, _ in self.rank_pages(question, **kw):
                head = count(p.text([]))
                if used + head > budget:
                    break
                used += head
                kept = []
                for b in p.blocks:
                    c = count(b + "\n")
                    if used + c > budget:
                        break
                    kept.append(b)
                    used += c
                out[p.url] = (p, kept)
                if used >= budget:
                    break
        elif mode == "chunks":
            for p, b, _ in self.rank_blocks(question):
                c = count(b + "\n") + (0 if p.url in out else count(p.text([])))
                if used + c > budget:
                    continue
                used += c
                out.setdefault(p.url, (p, []))[1].append(b)
        else:
            raise ValueError(mode)
        return list(out.values())


SCREEN_HEADER = re.compile(
    r'^\[(?P<clock>[^\]]*)\] Google Chrome(?:, tab "(?P<title>.*)" \((?P<url>\S*)\)(?: \(scrolled down\))?\.'
    r" Open tabs: .*|: New Tab page.*)$")


def split_header(screen_text: str) -> tuple[str | None, str | None, str]:
    """Benchmark screens start with a line naming the active tab (the window title / URL bar) and the tab strip.
    Returns (url, title, page text). The tab strip is dropped: it names other pages, which would make this
    screen match questions about them."""
    head, _, body = screen_text.partition("\n")
    m = SCREEN_HEADER.match(head)
    if not m:
        raise ValueError(f"unrecognized screen header: {head[:120]}")
    return m["url"], m["title"], body
