"""Parse the EUR-Lex HTML of the AI Act and the GDPR into citable chunks.

One chunk is a whole article, recital or annex, or, when longer than
MAX_CHARS, consecutive paragraphs of it. Every chunk carries a stable id of
the form ``<doc>/<kind>-<n>#<part>`` (e.g. ``ai-act/art-6#1``) and the
article-level ``unit`` (``ai-act/art-6``), which is what citations resolve to.
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

MAX_CHARS = 1800
DOCS = {
    "ai-act": {"file": "ai-act.html", "title": "Regulation (EU) 2024/1689 (AI Act)"},
    "gdpr": {"file": "gdpr.html", "title": "Regulation (EU) 2016/679 (GDPR)"},
}
# Top-level structural markers; a unit runs from its marker to the next one.
_MARK = re.compile(r'<div[^>]*\bid="((?:art|rct|anx|cpt|enc)_[A-Za-z0-9]+)"')
_UNIT = re.compile(r"^(art|rct|anx)_([A-Za-z0-9]+)$")


class _Text(HTMLParser):
    """Collect paragraph texts; a bare list marker like '(a)' joins the next one."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paras: list[str] = []
        self._buf: list[str] | None = None
        self._pending = ""

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self._buf = []

    def handle_data(self, data):
        if self._buf is not None:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag == "p" and self._buf is not None:
            t = re.sub(r"\s+", " ", "".join(self._buf).replace("\xa0", " ")).strip()
            self._buf = None
            if not t:
                return
            if re.fullmatch(r"\(?[0-9a-zA-Z]{1,5}[.)]", t) or re.fullmatch(r"\([0-9a-zA-Z]{1,5}\)", t):
                self._pending = (self._pending + " " + t).strip()
                return
            self.paras.append((self._pending + " " + t).strip())
            self._pending = ""


def _paras(fragment: str) -> list[str]:
    p = _Text()
    p.feed(fragment)
    return p.paras


def _split(paras: list[str], head: str) -> list[str]:
    """Group paragraphs into parts of at most MAX_CHARS (a long paragraph stays whole)."""
    parts, cur = [], []
    size = 0
    for para in paras:
        if cur and size + len(para) > MAX_CHARS:
            parts.append(cur)
            cur, size = [], 0
        cur.append(para)
        size += len(para) + 1
    if cur:
        parts.append(cur)
    return ["\n".join(p) for p in parts]


def parse_doc(doc: str, html: str) -> list[dict]:
    marks = [(m.start(), m.group(1)) for m in _MARK.finditer(html)]
    chunks: list[dict] = []
    for i, (pos, mid) in enumerate(marks):
        um = _UNIT.match(mid)
        if not um:
            continue
        kind, num = um.groups()
        end = marks[i + 1][0] if i + 1 < len(marks) else len(html)
        paras = _paras(html[pos:end])
        if not paras:
            continue
        # drop the trailing signature block of the regulation
        paras = [
            p for p in paras
            if not p.startswith(("Done at ", "For the European Parliament"))
            and not re.match(r"^\(\d+\) .*\(OJ L", p)  # OJ footnotes
        ]
        if kind == "art":
            # paras[0] = 'Article N', paras[1] = title
            title = paras[1] if len(paras) > 1 else ""
            body = paras[2:]
            label = f"Article {num} - {title}"
        elif kind == "anx":
            title = paras[1] if len(paras) > 1 else ""
            body = paras[2:] if len(paras) > 2 else paras[1:]
            label = f"Annex {num} - {title}".rstrip(" -")
        else:
            body = paras
            label = f"Recital {num}"
        if kind == "rct":
            body = [re.sub(r"^\(\d+\)\s*", "", b, count=1) for b in body]
        unit = f"{doc}/{kind}-{num}"
        texts = _split(body, label) or [""]
        for n, t in enumerate(texts, 1):
            chunks.append({
                "id": f"{unit}#{n}",
                "unit": unit,
                "doc": doc,
                "kind": kind,
                "label": label,
                "part": n,
                "parts": len(texts),
                "text": t,
            })
    return chunks


def build(raw_dir: Path, out: Path) -> list[dict]:
    allc: list[dict] = []
    for doc, meta in DOCS.items():
        html = (raw_dir / meta["file"]).read_text(encoding="utf-8")
        allc += parse_doc(doc, html)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in allc:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return allc


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def embed_text(c: dict) -> str:
    """What is embedded and BM25-indexed: the label plus the text."""
    return f"{c['label']}\n{c['text']}"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    cs = build(root / "data" / "raw", root / "data" / "processed" / "chunks.jsonl")
    from collections import Counter
    print(Counter((c["doc"], c["kind"]) for c in cs), len(cs), "chunks")
