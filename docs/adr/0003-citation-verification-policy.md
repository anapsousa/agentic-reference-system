# ADR 0003: an unverifiable quote is dropped, refused or escalated, never shipped

Status: accepted

Rule set (implemented in `verify` and `decide`):
1. A citation counts only if the chunk was in the candidate set shown to the model AND the quote is verbatim in it (P1 `verify_citation`). The first condition stops a model citing real law it was never shown (memory, not grounding).
2. Unverified obligations are removed and the answer says how many were dropped.
3. If the tier requires obligations (not minimal) and none verify: status `refused`.
4. If the tier's own supporting quote fails: the tier is withheld from the output and the case is `escalated`.
5. Low confidence, borderline, or an undetermined or unparseable classification: `escalated`, with the verified draft shown to the reviewer.
6. The answer text is rendered by code from verified items only. Model prose never reaches the output unfiltered, so there is no path for an unchecked quote.

Known gap (honest). This verifies the quote, not the claim built on it. The obligation `text` is the model's paraphrase and can overreach its quote (seen once in the smoke run: a "six months" detail the quote does not contain). Closing it needs an entailment check or showing only the quote; deferred to P3/P5 and recorded in the README.

Injection note. The description is passed as delimited data and the prompts say so, but that is a prompt-level mitigation, untested against attack. Indirect injection through retrieved text is P5.
