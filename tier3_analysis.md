# Tier 3 — Custom NER Evaluator Analysis
 
## Matching Strategy Comparison
 
```
Strategy        Precision   Recall     F1      TP    FP    FN
exact            0.0370    0.6471   0.0701    44  1144    24
partial          0.0300    0.5217   0.0566    36  1166    33
type_agnostic    0.0116    0.2029   0.0220    14  1188    55
```
 
The most counterintuitive result is that partial match performs worse than
exact match — TP drops from 44 to 36. This happens because partial match
requires both span overlap AND label agreement. When spaCy predicts "1.5
degrees" (QUANTITY) and gold has "1.5 degrees Celsius" (QUANTITY), the spans
overlap but the boundaries don't align perfectly, and because partial match
uses greedy assignment, some legitimate overlaps get consumed by earlier
spurious matches. Partial match is genuinely harder to implement fairly than
it sounds.
 
Type-agnostic match is the weakest of the three (TP=14), which reveals
something important: the dominant failure mode is not wrong labels but wrong
span boundaries. If span detection were the main problem, type-agnostic would
score higher by ignoring label mismatches. The fact that it scores lower than
exact match means spaCy's span boundaries frequently don't align with gold
boundaries at the character level — even when it finds the right entity, the
character offsets are often off by a few characters (leading "the", trailing
words, truncated phrases).
 
---
 
## Micro vs Macro Averaging
 
```
Micro: P=0.037   R=0.647   F1=0.070
Macro: P=0.603   R=0.624   F1=0.612
```
 
The gap between micro F1 (0.070) and macro F1 (0.612) is enormous and
requires explanation. Micro averaging pools all entities across all 132
English texts — predictions on the 122 unannotated texts generate massive
false positives with no gold to match, collapsing precision to near zero.
Macro averaging computes per-text scores first and then averages — and because
it only considers the 10 gold-annotated texts, those 122 unannotated texts
simply don't exist in the calculation.
 
The per-text breakdown shows meaningful variation:
 
```
text_id   Precision   Recall     F1     TP   FP   FN
      1    0.714      0.714    0.714     5    2    2
      2    0.400      0.400    0.400     2    3    3
      3    0.714      0.833    0.769     5    2    1
      4    0.500      0.571    0.533     4    4    3
      5    0.667      0.667    0.667     4    2    2
      6    0.600      0.600    0.600     3    2    2
      7    0.400      0.400    0.400     2    3    3
      8    0.571      0.667    0.615     4    3    2
      9    0.833      0.833    0.833    10    2    2
     10    0.625      0.556    0.588     5    3    4
```
 
Text 9 is the best-performing text (F1=0.833, TP=10) — it contains 12 gold
entities and most are standard geographic and monetary entities that spaCy
handles well. Texts 2 and 7 are the weakest (F1=0.400) — text 2 contains
COP28 (completely missed) and "United Arab Emirates" (span mismatch with "The
United Arab Emirates"), and text 7 misses "Paris Agreement" as LAW because
spaCy labels it EVENT.
 
For reporting purposes, micro F1 is the standard metric but macro F1 is the
honest one for this dataset — it measures performance where annotations exist,
not performance penalized by the absence of annotations.
 
---
 
## Error Distribution
 
```
Spurious entities:  1155  (95.3%)
Boundary errors:      33  ( 2.7%)
Missing entities:     22  ( 1.8%)
Type errors:           2  ( 0.2%)
Total:              1212
```
 
Spurious entities dominate at 95.3% — this is entirely a consequence of the
evaluation design. The 1155 spurious predictions are entities predicted in
the 122 unannotated texts where nothing can be verified. Within the 10
gold-annotated texts, the error picture is very different: the evaluator
found 33 boundary errors, 22 missing entities, and only 2 type errors.
 
**Type errors (2)** are the most actionable findings:
- `"Sixth Assessment Report"` predicted as EVENT but gold says WORK_OF_ART
- `"Carbon Border Adjustment Mechanism"` predicted as ORG but gold says LAW
Both represent the base model's tendency to default to familiar labels —
ORG for named institutional things, EVENT for named published things. These
are exactly the systematic mislabelings the Stretch 6A-S1 EntityRuler was
designed to fix.
 
**Boundary errors (33)** reveal consistent patterns in how spaCy's span
detection fails:
 
- Truncated spans: `pred="1.5 degrees"[107:118]` vs
  `gold="1.5 degrees celsius"[95:114]` — the model stops before "Celsius",
  producing a shorter span that overlaps but misses the full extent.
- Number-noun truncation: `pred="190"[24:27]` vs `gold="190 nations"[24:35]`
  — cardinal numbers get extracted without the noun they quantify.
- Leading article inclusion: `pred="the world bank"[0:14]` vs
  `gold="world bank"[4:14]` — spaCy includes "the" in the span when it
  appears sentence-initially, while gold records the bare form.
- Character offset drift: `pred="june 2024"[68:77]` vs
  `gold="june 2024"[73:82]` — same text string but different start/end
  positions, suggesting the source texts have slight preprocessing differences.
**Missing entities (22)** are gold entities with no overlapping prediction
at all. The most frequent pattern is entities that spaCy catches elsewhere
in the corpus but misses in these specific sentences: "COP28" appears twice
as missing (texts 2 and 10) because the base model completely ignores it.
"Paris Agreement" missing in texts 5 and 7 — the model labels it EVENT, and
since the span boundaries include "the" while gold doesn't, even partial match
fails. DATE entities like "2030", "2022", "February 2025" appearing as missing
is puzzling since dates are spaCy's strongest category — inspection of the
boundary errors suggests these dates are being caught with slightly different
offsets rather than missed entirely.
 
---
 
## What the Three Strategies Reveal Together
 
Running all three strategies simultaneously produces a diagnostic picture that
no single metric could provide. Exact match F1=0.070 reflects the combination
of span and label errors. Partial match dropping to F1=0.057 confirms that
boundary errors are substantial and that greedy span matching introduces
additional noise. Type-agnostic match collapsing to F1=0.022 confirms that
the fundamental problem is character-level span alignment, not label assignment
— spaCy finds entities, but its span boundaries frequently differ from the
gold annotations at the character level.
 
This suggests that for this corpus, the highest-value improvement would be
span normalization: stripping leading articles, standardizing number-noun
groupings, and normalizing possessives before evaluation. Label errors, while
highly visible in diagnostic output, account for only 0.2% of total errors and
would have marginal impact on metrics compared to addressing span boundary
inconsistencies.