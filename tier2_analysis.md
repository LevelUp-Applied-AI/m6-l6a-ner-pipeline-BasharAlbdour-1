# Tier 2 — Entity Aggregation Pipeline Analysis
 
## Normalization
 
The normalization step merged 34 surface forms into canonical entities and
removed 70 noise tokens. The most impactful merges were geographic and
institutional variants that spaCy produces inconsistently depending on
sentence position: `"the European Union"`, `"The European Union"`, and `"EU"`
all became `"European Union"`; `"the Middle East"` and `"The Middle East"`
(capitalization varying by sentence position) merged into `"Middle East"`,
collapsing two separate nodes that had been artificially splitting co-occurrence
counts. `"Jordanian"` merging into `"Jordan"` pushed Jordan to the most frequent
entity overall (26 occurrences), reflecting the dataset's strong MENA focus.
 
Noise removal was equally important. spaCy's DATE and CARDINAL taggers
misclassify temporal adverbs (`"annually"`, `"recent"`) and vague time spans
(`"50 years"`, `"20-year"`) as named entities. Before filtering, `Jordan +
annually` was the top co-occurring pair with 7 co-occurrences — a meaningless
artifact. After removing 70 such tokens, the top pairs became genuinely
informative: `2030 + Jordan = 5`, `2023 + 2030 = 5`, `Middle East + North
Africa = 4`.
 
---
 
## Co-occurrence Network
 
The network graph reveals two distinct structural patterns. First, temporal
hubs: `2023` (degree 7, the largest node) and `2030` (degree 6) connect to
almost every other entity in the graph. This is expected — climate texts
consistently reference current-year data (2023) and the 2030 emissions target
deadline, so these dates naturally co-occur with every named entity. They are
high-frequency connectors rather than thematically meaningful pairs.
 
More interesting are the geographic clusters. `Jordan + Middle East = 4`,
`Jordan + Saudi Arabia = 3`, `Middle East + North Africa = 4`, and `Middle
East + South Asia = 3` form a tight regional cluster in the right half of the
graph. This reflects the dataset's focus on MENA climate vulnerability — texts
about Jordan's water stress, regional heat projections, and cross-border
adaptation consistently name these geographic entities together. The
`Israel + Jordan = 2` pair is particularly notable: these two countries appear
together in texts discussing shared water resources in the Jordan Valley and
the Dead Sea, a context-specific co-occurrence that reflects real geopolitical
interdependence around climate adaptation.
 
The `2100 + Paris Agreement = 2` edge connects the agreement's long-term
temperature targets to end-of-century projections, appearing in science texts
that model warming scenarios under different policy pathways. `South Asia +
Sub-Saharan Africa = 2` reflects texts that group developing-region
vulnerability together when discussing adaptation finance and loss and damage.
 
The total of 4,451 unique entity pairs (down from 5,030 before normalization
and noise filtering) shows that cleaning meaningfully reduced spurious
connections while preserving real signal.
 
---
 
## TF-IDF Entity Importance
 
TF-IDF scores expose what makes each category linguistically distinctive beyond
simple frequency.
 
**Policy** is dominated by `Paris Agreement` with a TF-IDF of 0.031 — the
highest score of any entity in any category, and roughly double the next
highest entry (`$1.3 trillion` at 0.016). This makes sense: the Paris Agreement
is the central reference point for climate policy discourse, appearing in 6
policy texts while being absent or rare elsewhere. `Glasgow` (0.010) and
`Germany` (0.010) point to specific COP26 negotiations and European Green Deal
discussions. The cluster of monetary entities (`$1.3 trillion`, `$250 million`)
reflects the policy category's unique focus on climate finance commitments.
 
**Science** texts are distinctive for their measurement infrastructure and
polar geography: `Arctic` (0.014), `NASA` (0.014), `Red Sea` (0.014),
`GRACE-FO` (0.009), and `Antarctic` (0.009). GRACE-FO is a NASA satellite
mission for measuring ice sheet mass — its appearance signals highly specific
scientific source material. The presence of `"Science"` as an entity (TF-IDF
0.009) is a spaCy artifact: the word "Science" appears in journal citations
like "Nature Climate Science" and gets tagged as ORG, a known failure mode
for citation-heavy scientific text.
 
**Adaptation** texts are most distinctive for specific MENA implementation
sites: `Amman` (0.009), `Mafraq` (0.009), `Masdar City` (0.009). These are
not regional labels but specific cities and projects — Amman's water management
programs, Mafraq's refugee-affected agricultural zones, Masdar City's renewable
energy showcase in Abu Dhabi. This granularity distinguishes adaptation texts
from impact texts, which tend to name regions and countries rather than
specific sites.
 
**Impact** texts are characterized by the most vulnerable nations: `Libya`,
`Somalia`, `Turkey`, `Tuvalu` (all at 0.011). These are countries that
experienced specific recent climate disasters — Libya's 2023 floods, Somalia's
drought-famine crisis, Tuvalu's sea-level emergency. The appearance of `The
World Health Organization` as a distinctive impact entity (0.011) reflects
texts that link climate impacts to public health consequences.
 
---
 
## Summary
 
The aggregation pipeline revealed three things that raw entity counting misses.
First, normalization is non-trivial even for a clean dataset — 34 surface form
variations were present, and without merging them, co-occurrence analysis
produces artificially fragmented results. Second, the co-occurrence network
exposes the dataset's geographic center of gravity: this corpus is more focused
on MENA climate vulnerability than its category labels suggest. Third, TF-IDF
confirms that each category has a genuinely distinct linguistic signature —
policy texts are about the Paris Agreement and climate finance, science texts
are about measurement and polar regions, adaptation texts name specific
implementation sites, and impact texts name specific disaster-affected nations.
These distinctions would be invisible to a model that treats all climate text
as a single domain.