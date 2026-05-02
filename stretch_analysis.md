## Stretch Analysis — Multilingual NER Comparison

### What entity types are harder in Arabic vs. English, and why

The comparison table makes the cross-lingual gap immediately visible. spaCy's
xx_ent_wiki_sm found only 20 entities in 20 Arabic texts (density: 2.15 per 100
words) compared to 93 in the same number of English texts (density: 7.62 per 100
words) — a 3.5x drop. More telling than the count difference are the specific
errors: spaCy labeled the Arabic phrase وأكد التقرير ("the report confirmed") and
وأكد وزير ("the minister confirmed") as PER entities. These are verb phrases, not
person names. The model is pattern-matching on surface features rather than
understanding Arabic morphology, producing label noise instead of meaningful
extraction. Five Arabic texts returned zero entities entirely. The HF XLM-R model
tells a different story — 64 entities found, zero empty texts, and density of 6.88
per 100 words, close to its English performance. Crucially, it correctly identified
الهيئة الحكومية الدولية المعنية بتغير المناخ (the IPCC's full Arabic name) as ORG,
الأردن (Jordan) as LOC, and البنك الدولي (World Bank) as ORG. The core reason for
this gap is that Arabic NER is structurally harder than English NER: Arabic has no
capitalization to signal proper nouns, its morphological richness means entity
boundaries are harder to detect (words agglutinate prefixes and suffixes), and
diacritics that aid disambiguation are often absent in running text. A model like
xx_ent_wiki_sm, while multilingual, does not have sufficient Arabic training data
or architecture depth to handle these challenges. XLM-RoBERTa, trained on a much
larger multilingual corpus with a deeper transformer architecture, handles Arabic
morphology more robustly.

### What this means for bilingual NLP applications in the MENA region

In Jordan's professional environment, the same NLP pipeline routinely encounters
English technical reports, Arabic news articles, and mixed-language documents where
entities like "الأردن" and "Jordan" or "البنك الدولي" and "World Bank" refer to
the same real-world entity. This stretch demonstrates that model choice is not
neutral in a bilingual context — deploying spaCy xx_ent_wiki_sm in a production
MENA pipeline would silently produce low-quality Arabic extractions while appearing
to work correctly on English, creating an invisible quality gap that could mislead
downstream analysis. The HF XLM-R model is a more reliable foundation for bilingual
pipelines, but it introduces its own tradeoffs: it is significantly slower, requires
more memory, and uses a reduced label schema (PER/LOC/ORG only, no DATE or PERCENT)
compared to English-only spaCy models. A practical production approach for MENA
bilingual NLP would route English text to a domain-fine-tuned English model for
richer entity type coverage, and Arabic text to XLM-R or a purpose-built Arabic NER
model such as CAMeL-NER, combining their outputs through a unified entity resolution
layer that can match "الأردن" to "Jordan" across languages. The entity density gap
between English (7.62) and Arabic (6.88) under XLM-R also suggests that even the
better model leaves some Arabic entities undetected — a consideration for any system
where Arabic coverage is critical to the analysis.