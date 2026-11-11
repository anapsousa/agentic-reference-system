"""Deterministic paraphrase-consistency gate (ADR 0005).

Extracts the hard facts from a model's paraphrase (numbers, durations, percentages, amounts,
article/annex references, modal strength) and requires each to appear, normalised, in the
verified quote or in the cited chunk. No model is involved."""
from __future__ import annotations

import re

_UNITS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
          9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
          16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"}
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
# EN plus the Portuguese forms a pt-PT paraphrase would use
_PT = {"um": 1, "uma": 1, "dois": 2, "duas": 2, "três": 3, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6,
       "sete": 7, "oito": 8, "nove": 9, "dez": 10, "doze": 12, "quinze": 15, "vinte": 20, "trinta": 30}
_WORDS = {w: n for n, w in {**_UNITS, **_TENS}.items()} | _PT
_UNIT_STEM = {"month": "month", "months": "month", "mês": "month", "meses": "month", "mes": "month",
              "day": "day", "days": "day", "dia": "day", "dias": "day",
              "hour": "hour", "hours": "hour", "hora": "hour", "horas": "hour",
              "week": "week", "weeks": "week", "semana": "week", "semanas": "week",
              "year": "year", "years": "year", "ano": "year", "anos": "year"}
_STRONG = re.compile(r"\b(shall|must|is required|are required|required to|obliged|have to|has to|deve|devem)\b", re.I)
_WEAK = re.compile(r"\b(may|can|could|might|is permitted|are permitted|pode|podem)\b", re.I)
_ROMAN = r"[ivxlc]+"
_REF = re.compile(rf"\b(art(?:icle|igo)?s?\.?\s*(\d+)|annex(?:e)?\s+({_ROMAN})\b)", re.I)
_NUM_WORD = re.compile(r"\b(" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")(?:[- ](one|two|three|four|five|six|seven|eight|nine))?\b", re.I)


def _words_to_digits(s: str) -> str:
    def sub(m: re.Match) -> str:
        w1 = m.group(1).lower()
        n = _WORDS[w1]
        if m.group(2):
            if n % 10 or n < 20:      # "ten one" is not a number
                return m.group(0)
            n += _WORDS[m.group(2).lower()]
        return str(n)
    s = re.sub(r"\ba hundred\b|\bhundred\b", "100", s, flags=re.I)
    return _NUM_WORD.sub(sub, s)


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace(" ", " ").lower()
    s = _words_to_digits(s)
    s = re.sub(r"(\d)\s*(?:per cent|percent)\b", r"\1 %", s)
    s = re.sub(r"(\d)\s*%", r"\1 %", s)
    s = re.sub(r"\b(\d+(?:[.,]\d+)?)\s*(million|milhões|milhao|milhão|billion)\b",
               lambda m: str(int(float(m.group(1).replace(",", ".")) * (10**6 if m.group(2)[:3] in ("mil",) else 10**9))), s)
    s = re.sub(r"(?<=\d)[ ,.](?=\d{3}\b)", "", s)          # 35 000 000 / 35,000,000 -> 35000000
    s = re.sub(rf"\b(artigos?|articles?|art\.?)\s*(\d+)", r"art \2", s)
    s = re.sub(rf"\b(annexe?|anexo)\s+({_ROMAN})\b", r"annex \2", s)
    s = re.sub(r"\b(\d+)\s+([a-zçê]+)", lambda m: f"{m.group(1)} {_UNIT_STEM.get(m.group(2), m.group(2))}", s)
    return re.sub(r"\s+", " ", s)


def _facts(paraphrase: str) -> list[tuple[str, str]]:
    t = _norm(paraphrase)
    out: list[tuple[str, str]] = []
    for m in re.finditer(r"\bart (\d+)", t):
        out.append(("ref", f"art {m.group(1)}"))
    for m in re.finditer(rf"\bannex ({_ROMAN})\b", t):
        out.append(("ref", f"annex {m.group(1)}"))
    t = re.sub(r"\bart \d+(\(\d+\))?|\bannex " + _ROMAN + r"\b|\(\w{1,3}\)", " ", t)  # refs and "(a)" markers
    for m in re.finditer(r"\b(\d+) %", t):
        out.append(("percent", f"{m.group(1)} %"))
    t = re.sub(r"\b\d+ %", " ", t)
    for m in re.finditer(r"\b(\d+) (month|day|hour|week|year)\b", t):
        out.append(("duration", f"{m.group(1)} {m.group(2)}"))
    t = re.sub(r"\b\d+ (month|day|hour|week|year)\b", " ", t)
    for m in re.finditer(r"\b\d+\b", t):
        out.append(("number", m.group(0)))
    if _STRONG.search(paraphrase):
        out.append(("modal", "strong"))
    if _WEAK.search(paraphrase):
        out.append(("modal", "weak"))
    return list(dict.fromkeys(out))


def check_consistency(paraphrase: str, quote: str, chunk: dict) -> tuple[list[str], list[str]]:
    """Return (missing, outside_quote). `missing`: facts in neither quote nor chunk, so the
    obligation must be dropped. `outside_quote`: facts supported by the chunk but not the quote."""
    q, c = _norm(quote), _norm(f"{chunk['label']} {chunk['text']}")
    own = re.search(r"-(art|anx)-(\w+)#", chunk["id"])           # a chunk may cite itself
    if own:
        c += f" {'art' if own.group(1) == 'art' else 'annex'} {own.group(2).lower()}"
    missing, outside = [], []

    def has(src: str, kind: str, v: str) -> bool:
        if kind == "modal":
            return bool((_STRONG if v == "strong" else _WEAK).search(src))
        return re.search(rf"(?<![\w.]){re.escape(v)}(?![\w])", src) is not None

    for kind, v in _facts(paraphrase):
        if has(q, kind, v):
            continue
        (outside if has(c, kind, v) else missing).append(f"{kind}:{v}")
    return missing, outside
