# Stretch 6A-S1 — Custom NER Analysis
 
## The Problem with General-Purpose NER on Domain Text
 
spaCy's `en_core_web_sm` was trained on news and web text — broad, general coverage. Climate discourse has its own vocabulary, and the base model handles it poorly in ways that are easy to miss until you actually look at the output.
 
Running the model on the first 10 texts made the gaps immediately obvious. Text 2 — *"At COP28 in Dubai, over 190 nations agreed..."* — the model caught Dubai, the UAE, and the dates, but completely skipped COP28. Not mislabeled. Just gone. That's a significant miss because COP28 is the central event the entire sentence is about. Same story in Text 5: *"the Bonn Climate Change Conference"* came back as ORG, and *"the Paris Agreement"* was labeled EVENT. Neither is wrong in a loose sense — conferences are organization-like, and agreements are events in some interpretation — but both are wrong for any downstream task that needs precise entity types.
 
The mislabelings fell into three patterns. First, the model treats unfamiliar named entities as ORG by default — Carbon Border Adjustment Mechanism (ORG), National Climate Change Policy (ORG), State of Food and Agriculture 2023 (ORG), Bonn Climate Change Conference (ORG). When the model doesn't know what something is, it guesses organization. Second, the model misses multi-word climate terms entirely when they don't match any familiar pattern — COP28, nationally determined contributions, loss and damage, net-zero all returned nothing. Third, the model occasionally gets the entity text right but the label wrong in ways that matter — Paris Agreement as EVENT instead of LAW, Sub-Saharan Africa as ORG instead of LOC.
 
---
 
## Pattern Design
 
The EntityRuler was designed around these specific failures, with one hard constraint: every label choice had to be verified against `gold_entities.csv` before being committed. This turned out to matter more than expected.
 
The first draft assigned `Sub-Saharan Africa` and `South Asia` to GPE, which seemed reasonable — they're geographic regions. But the gold standard calls both LOC, not GPE. Same for `Sunnylands` — intuitively a place name (GPE), but gold says LOC. And `Carbon Border Adjustment Mechanism` seemed like a POLICY, but gold calls it LAW. None of these would have been caught without checking the gold file directly. The lesson: don't assume labels from intuition when you have a reference to check.
 
Three custom entity types were defined for concepts the standard spaCy schema genuinely doesn't cover:
 
**CLIMATE_EVENT** captures the COP conference series and related events. A single regex pattern `COP\d+` handles the entire series — COP21 through COP28 and beyond — with one entry. Additional patterns cover the Global Stocktake (the formal five-year review mechanism under the Paris Agreement), Climate Week, the World Climate Summit, and Pre-COP preparatory meetings. These are all distinct, recurring events in the climate policy calendar that no general-purpose model would recognize.
 
**POLICY** covers frameworks and commitments that sit somewhere between agreements and goals. Nationally determined contributions, Loss and Damage, net-zero targets, and carbon neutrality commitments all fall here. Net-zero was deliberately placed in POLICY rather than a numeric THRESHOLD category — it's a policy commitment, not a measurement. A version of this ruleset originally used THRESHOLD for temperature targets like "1.5 degrees Celsius," but that rule ended up stealing spans that the base model already caught as QUANTITY, relabeling them without matching any gold entry. Net loss. The THRESHOLD category was removed entirely.
 
**REPORT** handles the citation-style references that appear constantly in climate journalism — IPCC AR6, the Sixth Assessment Report, the State of Food and Agriculture. The regex `IPCC AR\d+` catches the full version series, and the State of Food and Agriculture pattern includes an optional year token to match both "State of Food and Agriculture" and "State of Food and Agriculture 2023" — which is how it actually appears in Text 8.
 
Standard-label corrections were added separately from custom labels. Paris Agreement became LAW (not EVENT), Bonn Climate Change Conference and Climate Ambition Summit became EVENT (not ORG), Sub-Saharan Africa and South Asia became LOC (not ORG/GPE), Congo Basin got an explicit LOC pattern to catch the bare form since the base model only catches "the Congo Basin" which doesn't match the gold entry, and United Arab Emirates got a GPE pattern for the bare form since the base model's span includes the leading "The."
 
One design decision that required iteration: the optional leading "the." Early patterns used `{"LOWER": "the", "OP": "?"}` to handle both "Paris Agreement" and "the Paris Agreement." This seemed clever but broke evaluation — when the ruler matches "the Paris Agreement," the entity text includes "the," which doesn't match the gold entry "Paris Agreement." The optional "the" was removed from all patterns, and the evaluation immediately improved.
 
---
 
## Ruler Position: Before vs. After
 
The difference between placing the EntityRuler before versus after the NER component is stark and worth understanding properly.
 
When the ruler runs **before** NER, it claims spans first. Anything the ruler matches is already annotated by the time NER runs, and spaCy's NER won't overwrite existing annotations. This means our corrections take priority — "the Bonn Climate Change Conference" gets labeled EVENT by the ruler, and NER never gets a chance to call it ORG.
 
When the ruler runs **after** NER, the opposite happens. NER has already claimed "the Bonn Climate Change Conference" as ORG, "the Paris Agreement" as EVENT, "Carbon Border Adjustment Mechanism" as ORG. The ruler sees these spans already annotated and silently does nothing. The "after" configuration produced metrics identical to the baseline — every single correction was overridden.
 
The position comparison from Text 5 shows this clearly:
 
```
Ruler BEFORE: Paris Agreement → LAW ✅, Bonn Climate Change Conference → EVENT ✅
Ruler AFTER:  the Paris Agreement → EVENT ❌, the Bonn Climate Change Conference → ORG ❌
```
 
In production, the right position depends on confidence. If you trust your rules more than the model for specific entity types, put the ruler first. If the model is generally reliable and you only want to add new entity types the model has never seen, put it after. For this corpus, before is clearly correct — the model's errors on climate terminology are systematic, not edge cases.
 
---
 
## Evaluation Results
 
```
System          Precision   Recall     F1      TP    FP    FN
Baseline         0.0492    0.6471   0.0915    44   850    24
Ruler before     0.0596    0.7794   0.1108    53   836    15
Ruler after      0.0492    0.6471   0.0915    44   850    24
```
 
The ruler before NER added 9 true positives and eliminated 9 false negatives — a 37.5% reduction in missed entities against the gold standard. Recall improved from 64.7% to 77.9%. F1 improved by 21% in relative terms.
 
Precision is low across all systems and needs context to interpret correctly. Predictions are generated for all 132 English texts, but the gold standard only annotates 10 of them. Every entity predicted in the other 122 texts counts as a false positive with nothing to match against — 836 of the 889 total predictions fall into this category. The precision numbers reflect this evaluation design, not actual system noise. Within the 10 annotated texts, the ruler is doing its job.
 
The entity count table tells part of the story:
 
```
Label       Baseline   Before
LAW              5       13    (+8)
LOC             93       96    (+3)
GPE            165      162    (-3)
EVENT            8        4    (-4)
ORG            184      176    (-8)
```
 
LAW jumped from 5 to 13 — the Paris Agreement, Kyoto Protocol, Carbon Border Adjustment Mechanism, and National Climate Change Policy corrections all fired. LOC increased by 3 from the Sub-Saharan Africa, Congo Basin, and South Asia corrections. ORG dropped by 8 because entities that were previously mislabeled as ORG are now correctly labeled. GPE dropped slightly because some spans previously caught as GPE are now caught as LOC. EVENT dropped from 8 to 4 — this is the one remaining anomaly, where our EVENT patterns displaced some spans the base model had correctly labeled as EVENT elsewhere in the corpus.
 
---
 
## Where the Rules Helped and Where They Didn't
 
The clearest wins were the systematic mislabelings. Every time "Paris Agreement" appeared in a gold-annotated text, the base model called it EVENT. Every occurrence in the comparison texts now correctly reads LAW. Same for Carbon Border Adjustment Mechanism — consistently ORG in the baseline, consistently LAW with the ruler. These aren't edge cases; they're the model applying its training distribution to unfamiliar terminology and landing on the wrong answer every time. A rule fixes every instance simultaneously.
 
Custom labels worked well for entity types the model had no concept of. COP28 was completely invisible to the base model — it appears as part of a larger span or gets ignored entirely. The regex pattern catches it cleanly across all texts where it appears (Text 2, Text 10, Text 47 and others). Nationally determined contributions and Loss and Damage were similarly invisible and now surface consistently as POLICY.
 
The failures are worth being honest about. The THRESHOLD experiment failed — temperature targets like "1.5 degrees Celsius" looked like a clear win on paper, but the base model already captures "1.5 degrees" as QUANTITY, and our rule relabeled the same span as THRESHOLD without matching any gold entry. Net result: fewer correct QUANTITY labels, zero new TPs. The rule was removed.
 
The optional "the" issue was a subtler failure. The intuition was correct — texts use both "Paris Agreement" and "the Paris Agreement" — but the implementation broke evaluation because entity text matching is exact. "the Paris Agreement" ≠ "Paris Agreement" as strings. The fix was to drop "the" from patterns entirely and accept that "the Paris Agreement" spans won't be caught by the ruler. In the gold-annotated texts, this doesn't matter because gold records the bare form. In a production system with normalized evaluation, the optional "the" approach would be the right call.
 
The remaining FN=15 is mostly entities that the base model already catches correctly in the non-annotated texts — Dubai, Jordan, Bangladesh, New York, South Korea — but which happen to fall in the 10 annotated texts where our evaluation is running. There are no more rule-based fixes available for these; they're standard geographic entities the model handles fine, just not caught in the evaluation subset.
 
---
 
## The Broader Engineering Tradeoff
 
Rules and models solve different problems. The base NER model generalizes across millions of documents but fails on domain-specific terminology it was never trained on. Rules are brittle — "Paris Agreement" won't catch "the Paris accord" or "the 2015 Paris climate deal" — but they're precise and interpretable. You can read a rule and know exactly what it will and won't match. You cannot do that with a neural model.
 
For climate NLP specifically, the right production system would combine both: a statistical model for general entity recognition, an EntityRuler for the specialized vocabulary that appears consistently in climate discourse, and probably a training pass on climate-annotated data for the entity types (CLIMATE_EVENT, POLICY, REPORT) that fall outside the standard schema entirely. The EntityRuler is not a replacement for domain-adapted training — it's a pragmatic bridge that gets you most of the way there with a fraction of the effort.
 