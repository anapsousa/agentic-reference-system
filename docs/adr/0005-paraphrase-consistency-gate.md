# ADR 0005: deterministic paraphrase-consistency gate, not a model-based entailment check

Status: accepted (closes the gap named in ADR 0003, partly)

Context. `verify` proves the quote is real. The obligation text is the model's paraphrase and can assert more than the quote.

Decision. A `consistency` node after `verify`. From each paraphrase it extracts hard facts: plain numbers, durations (number plus month/day/hour/week/year), percentages, large amounts (35 000 000, 35 million), article and annex references, and modal strength (shall/must versus may). Each must appear, after normalisation (number words and digits, thousands separators, "percent", EN and PT units and numerals, "artigo"/"article"), in the verified quote or in the cited chunk. A fact in neither: the obligation is dropped with a recorded reason, and the existing rules apply if none remain (refuse). A fact in the chunk but not the quote passes and is recorded as `outside_quote` for the reviewer.

Why not a model check. It costs tokens on every obligation, it is itself non-deterministic, and its recall would need its own evaluation (the lab's judge reached 25% recall). A regex gate is free, repeatable and explainable.

Limits (honest).
- It checks facts, not meaning. "Providers must not do X" paraphrased as "Providers must do X" passes if the words and numbers are present. Negation and scope are not checked.
- Chunk-level support is a deliberate leniency: the fact may sit in another sentence of the chunk than the quote.
- Modal check is coarse (any "shall/must" in the source satisfies a strong claim).
- Numeric words are handled up to 99 plus hundred; Portuguese covers common numerals only.
- The smoke run's "at least six months" turned out to be TRUE (Art 26(6) is in the cited chunk); the earlier README note called it unsupported, which overstated it. It now passes with an `outside_quote` flag. A fabricated "twelve months" is dropped (tested).
