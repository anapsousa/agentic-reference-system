# ADR 0002: tier-defining and obligation-bearing articles are injected, not only retrieved

Status: accepted

Context. P1 found plain-language queries retrieve the right article about 3 times in 4 at rank 1, and Art 6 reaches Annex III only by cross-reference, which the retriever does not follow. A description such as "we rank job applicants" shares few words with Annex III point 4(a). Obligation articles (Arts 9 to 15) are not what a description is about, so they do not rank for it either.

Decision. (1) Always show the classifier Art 5, Art 6, Annex III and Art 50 beside the top-8 operative hits (recitals are excluded, per P1). (2) After the tier is known, add that tier's obligation articles to the citer's context (high: Arts 9 to 16 and 26; prohibited: Art 5; limited: Art 50). The candidate set is then fixed per case and recorded in state.

Cost. A larger prompt (about 34k cache-creation tokens in the smoke run). Acceptable at this scale; revisit if P3 shows cost dominating.

Found by test: the first happy-path test failed because BM25 never surfaced Art 9 for a CV-screening description. That is the evidence for (2).
