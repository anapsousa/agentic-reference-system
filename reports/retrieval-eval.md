# P1 retrieval evaluation

Run: 2026-11-10 | embedding model: `BAAI/bge-small-en-v1.5` (384d, cosine (flat, exact)) | chunks: 775 | gold questions: 44

| retriever | hit@1 | hit@3 | hit@5 | hit@10 | recall@1 | recall@3 | recall@5 | recall@10 | mrr | citation@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.32 | 0.57 | 0.70 | 0.82 | 0.30 | 0.53 | 0.66 | 0.77 | 0.47 | 0.27 |
| dense | 0.27 | 0.59 | 0.75 | 0.89 | 0.25 | 0.55 | 0.70 | 0.84 | 0.47 | 0.18 |
| hybrid | 0.30 | 0.61 | 0.75 | 0.93 | 0.27 | 0.56 | 0.70 | 0.87 | 0.49 | 0.25 |
| bm25+operative | 0.55 | 0.73 | 0.84 | 0.91 | 0.50 | 0.68 | 0.78 | 0.84 | 0.66 | 0.45 |
| dense+operative | 0.73 | 0.95 | 0.98 | 1.00 | 0.68 | 0.91 | 0.93 | 0.96 | 0.84 | 0.55 |
| hybrid+operative | 0.73 | 0.86 | 0.93 | 1.00 | 0.66 | 0.80 | 0.88 | 0.96 | 0.82 | 0.55 |

`+operative` = recitals excluded (articles and annexes only). Recitals paraphrase the law in plain
language and outrank the operative text on natural-language queries; an obligation must be cited from an article.

## Misses at rank 1 (hybrid+operative)

- q02 "Can a city government score citizens on their social behaviour and use that score to limit their access to services?" -> top1 `ai-act/anx-III#2`, wanted ['ai-act/art-5'], hit@5=1
- q03 "We want to detect employees' emotions through their webcam during work calls. Is that allowed?" -> top1 `ai-act/art-50#2`, wanted ['ai-act/art-5'], hit@5=0
- q04 "Is it lawful to build a facial recognition database by scraping photos from the internet or CCTV?" -> top1 `ai-act/art-50#2`, wanted ['ai-act/art-5'], hit@5=1
- q05 "When is an AI system classified as high-risk?" -> top1 `ai-act/art-80#1`, wanted ['ai-act/art-6'], hit@5=1
- q09 "What does a provider of a high-risk AI system have to put in place for risk management?" -> top1 `ai-act/art-25#1`, wanted ['ai-act/art-9'], hit@5=1
- q15 "What are the main obligations of a company that deploys (uses) a high-risk AI system but did not build it?" -> top1 `ai-act/art-25#1`, wanted ['ai-act/art-26'], hit@5=1
- q17 "Must users be told they are talking to a chatbot?" -> top1 `ai-act/art-31#1`, wanted ['ai-act/art-50'], hit@5=1
- q21 "What are the fines for breaching the AI Act's prohibited practices?" -> top1 `ai-act/art-100#2`, wanted ['ai-act/art-99'], hit@5=0
- q23 "When does the AI Act start to apply and what applies earlier?" -> top1 `ai-act/art-62#1`, wanted ['ai-act/art-113'], hit@5=0
- q27 "Must high-risk AI systems be registered in an EU database?" -> top1 `ai-act/art-71#1`, wanted ['ai-act/art-49'], hit@5=1
- q29 "Can people ask for an explanation of a decision made with a high-risk AI system?" -> top1 `ai-act/art-14#2`, wanted ['ai-act/art-86'], hit@5=1
- q33 "Is a decision made only by an algorithm, with legal effect on a person, permitted?" -> top1 `ai-act/art-86#1`, wanted ['gdpr/art-22'], hit@5=1
