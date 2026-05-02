# Tier 1 — Per-Category NER Analysis
 
## Entity Distribution by Category
 
DATE is the one entity type that appears consistently across all four categories
(62 in adaptation, 51 in impact, 69 in policy, 74 in science), which makes sense
— climate writing is inherently temporal regardless of angle. Beyond that, the
categories diverge sharply.
 
**Science** texts stand out for their heavy LOC (37) and QUANTITY (33) counts
relative to other categories. Scientific reporting tends to name specific
geographic regions being studied — the Arctic, the Congo Basin, the Great Barrier
Reef — and quantify measurements precisely. Science also has the highest ORG
count (57), driven by research institutions and intergovernmental bodies cited
as sources. Notably, MONEY is completely absent from science texts (0), which
reflects the difference between describing physical phenomena and discussing
funding or economic consequences.
 
**Policy** texts show the opposite pattern on MONEY (32 — highest of any
category) and have the only meaningful EVENT count (7 vs 0-1 elsewhere). This
makes intuitive sense: policy discourse centers on funding commitments, budget
allocations, and specific negotiation events like COP conferences and summits.
Policy also has the highest PERSON count (12), reflecting the named officials
and negotiators who appear in institutional coverage. LAW entities (3) appear
almost exclusively here, which aligns with what you'd expect — agreements,
protocols, and mechanisms are policy instruments.
 
**Adaptation** texts are the most entity-dense category overall (348 total),
but their entities are spread across generic types — DATE, GPE, ORG, CARDINAL,
PERCENT. The high GPE count (55) reflects that adaptation writing tends to
focus on specific countries and regions implementing measures. The high CARDINAL
and PERCENT counts (48 and 33) come from reporting on quantitative targets and
progress metrics.
 
**Impact** texts have the lowest total entity count (269) and the most balanced
distribution. TIME entities appear here more than anywhere else (5 vs 3/0/0),
which suggests impact writing references timeframes and durations — "over the
past decade," "within the next 20 years" — in ways other categories don't.
The low ORG count (27, vs 49-57 elsewhere) suggests impact texts focus more
on affected populations and regions than on institutional actors.
 
---
 
## NER Performance by Category
 
Gold annotations only cover **policy** texts (10 out of 43 policy texts are
annotated). The other three categories — adaptation, impact, science — have
no gold standard entries at all, making quantitative evaluation impossible
for them.
 
```
Category    Precision   Recall     F1      TP    FP    FN
policy       0.1606    0.6471   0.2573    44   230    24
adaptation      —         —        —       —     —     —   (no gold)
impact          —         —        —       —     —     —   (no gold)
science         —         —        —       —     —     —   (no gold)
```
 
Policy recall of 64.7% means spaCy finds roughly two-thirds of gold-annotated
entities in policy texts. Precision (0.16) is higher than the full-corpus
evaluation (0.05) because here we only count false positives within the 43
policy texts rather than across all 132 English texts.
 
---
 
## Easiest and Hardest Categories for NER
 
**Policy** is the easiest category for NER — not because it has simpler
language, but because it's densely populated with the exact entity types
spaCy was trained on: named organizations (IPCC, World Bank, UNEP), named
locations (Jordan, UAE, New York), specific dates, and named individuals.
These are the bread-and-butter entities of general-purpose NER training data,
which skews toward news and institutional text.
 
**Science** is the hardest in a different way. The entities are there — LOC,
QUANTITY, ORG — but they appear in contexts the model struggles with. A
sentence like "warming of 1.48 degrees Celsius above pre-industrial levels"
contains a QUANTITY that gets partially caught and a temporal reference that
gets missed. Scientific language is precise in ways that don't match the
model's training distribution.
 
**Adaptation** is likely the hardest for recall specifically. Adaptation texts
describe processes, behaviors, and interventions — "drought-resistant crop
varieties," "community-based water management," "early warning systems" — that
contain few canonical named entities. The model finds DATE, GPE, and ORG
entities where they exist, but a large portion of the meaningful content in
adaptation texts is not entity-shaped at all, meaning the model's output
represents only a fraction of what a human reader would consider informative.
 
The absence of gold annotations for adaptation, impact, and science means
these assessments are qualitative — grounded in the entity count distributions
rather than precision/recall numbers. A properly annotated gold standard
spanning all four categories would be needed to make these claims quantitatively.