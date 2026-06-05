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
# Stretch Analysis: Cross-Lingual Embedding Comparison
## Module 6 Week B — bert-base-multilingual-cased on Climate Text
 
---
 
## Text Selection Methodology
 
Before discussing the results, it is worth explaining how the 10 English/Arabic text pairs were selected — because the selection method is a deliberate design decision, not an afterthought.
 
The most obvious automated approach would have been TF-IDF-based pairing: extract TF-IDF vectors for all English and Arabic texts, compute cosine similarity across languages, and select the top-scoring pairs. This is appealing because it removes human judgment from the loop entirely. However, it has a critical flaw for this specific experiment — **circularity**. The goal of this assignment is to evaluate whether multilingual BERT embeddings can identify semantically similar content across languages. If we use a similarity metric to *select* our pairs and then use a *different* similarity metric to *evaluate* those same pairs, we are not testing the model on genuinely independent ground truth. We would be measuring how well BERT agrees with TF-IDF, not how well BERT captures cross-lingual meaning.
 
A second issue with TF-IDF across languages is that it operates on surface vocabulary. Arabic and English share almost no characters, so a raw TF-IDF similarity between an Arabic and an English document would be near zero regardless of topic — the only shared tokens would be numerals and Latin acronyms like "IPCC" or "COP28". This makes TF-IDF a poor cross-lingual similarity signal without translation or transliteration as a preprocessing step, neither of which belongs in a text selection utility.
 
The approach taken here is simpler and more transparent: for each of 10 named climate topics, a short keyword is defined for each language that unambiguously identifies articles on that topic. For example, the keyword `"36.8 billion tonnes"` in English and `"36.8 مليار طن"` in Arabic both point to the same Global Carbon Project statistic, guaranteeing that the matched pair covers the same event. This method is intentional and explainable — anyone reading the code understands exactly why each pair was selected — and it avoids the circularity problem entirely because the selection criterion (keyword presence) is completely independent of the evaluation criterion (embedding cosine similarity).
 
---
 
## Part (a): How Well Does the Multilingual Model Capture Cross-Lingual Similarity?
 
The results show that `bert-base-multilingual-cased` does capture meaningful cross-lingual structure in climate text, but the full picture only emerges when you look at all three similarity baselines together — cross-lingual same-topic, within-English, and within-Arabic — rather than the cross-lingual scores in isolation.
 
Starting with the cross-lingual results: across all 10 same-topic English/Arabic pairs, the mean cosine similarity was **0.7067**, compared to a mean off-diagonal similarity of **0.6127** — a gap of **0.094**. This gap is the core finding. Same-topic pairs are generally ranked higher on average than mismatched pairs across all 10 topics, which means the model's shared embedding space is doing real semantic work across languages. This is not a given. The model was never explicitly trained to align Arabic and English representations — it learned a shared space purely from multilingual co-occurrence patterns during pretraining on 104 languages simultaneously, with no cross-lingual supervision signal.
 
Looking at individual pairs, the scores tell a coherent story. The strongest same-topic pair was **Amman Heatwave** (EN-29 × AR-95) at **0.7646**. This makes intuitive sense: both texts refer to the same city (Amman / عمّان), the same temperature threshold (40°C), the same year (2023), and the same statistical claim about 18 days above that threshold. The model latches onto these shared numerical and named-entity anchors across languages, producing a high similarity score even though the surface text looks completely different. On the other end, **COP28 Dubai** (EN-2 × AR-81) scored only **0.6504** — the lowest same-topic score. Looking at the actual texts explains why: the English article emphasizes the UAE presidency brokering the fossil fuel transition deal, while the Arabic article focuses on developing nations demanding climate damage compensation. They cover the same event but through different narrative angles, and the model correctly reflects that thematic divergence with a lower score. This is the model working as intended — it is sensitive to content, not just topic labels.
 
The within-language baselines add critical context to these numbers. Within-English similarity averaged **0.7486** and within-Arabic similarity averaged **0.8059** — both higher than the cross-lingual same-topic mean of 0.7067. The within-Arabic score being the highest of the three is worth examining closely. The Arabic articles in this dataset tend to be shorter and more structurally uniform than their English counterparts, with less vocabulary variation across texts. This pulls Arabic embeddings closer together in the embedding space regardless of topic, inflating the within-AR baseline. It does not mean Arabic climate text is inherently more semantically coherent — it reflects a dataset characteristic rather than a linguistic one. The practical implication is that the model is not perfectly language-agnostic: Arabic and English occupy overlapping but not identical regions of the shared embedding space, and the gap between within-language and cross-lingual scores is the measurable cost of multilingual generalization.
 
One finding in the top-3 most similar pairs deserves specific attention: **EN-1 (IPCC Assessment Report) × AR-81 (COP28 Dubai)** scored **0.7516**, which is higher than nine of the ten actual same-topic diagonal scores. This is a cross-topic match outperforming most same-topic ones. The reason is domain overlap — both texts are dense with shared climate policy vocabulary: emissions reductions, 1.5 degrees Celsius, 2030 targets, fossil fuels, Paris Agreement. When two texts share this much domain-specific terminology across languages, the model's shared subword vocabulary picks up on overlapping concepts even across different events. This is simultaneously a strength and a limitation: the model generalizes well within a domain, but it struggles to cleanly separate closely related topics that share most of their domain vocabulary. The ranking is preserved on average — same-topic pairs score higher than random off-diagonal pairs — but individual topic boundaries are blurry at the edges of the domain.
 
---
 
## Part (b): What Does This Mean for Building Bilingual NLP Tools in the MENA Region?
 
The results from this experiment have concrete and mixed implications for anyone designing bilingual Arabic/English NLP systems for deployment in the Middle East and North Africa, and they argue for a nuanced deployment strategy rather than a blanket recommendation in either direction.
 
The encouraging finding is that a single multilingual model can serve as the backbone for bilingual search and retrieval without requiring separate per-language models or machine translation as a preprocessing step. The consistent 0.094 gap between same-topic and off-topic cross-lingual pairs means that a retrieval system built on these embeddings would surface topically relevant Arabic documents in response to English queries — and vice versa — more reliably than chance. For a climate information portal serving both Arabic and English readers in Jordan, Egypt, or the broader MENA region, this is operationally valuable. A journalist querying in English about Amman's heat crisis would retrieve the Arabic AR-95 article near the top of results, because the embedding space preserves the semantic connection at a similarity of 0.7646 — well above the off-diagonal mean of 0.6127. This reduces the cost of maintaining separate monolingual indices and eliminates the latency and error introduced by translation pipelines. For resource-constrained organizations operating in the region, deploying one multilingual model instead of two monolingual ones is a meaningful practical advantage.
 
However, three findings from this experiment argue for caution before deploying this model in production classification or ranking systems. First, the within-language gap: within-Arabic similarity averaged 0.8059 and within-English averaged 0.7486, both substantially above the cross-lingual same-topic mean of 0.7067. Any system that uses a single cosine similarity threshold for both within-language and cross-lingual comparisons will behave inconsistently — producing more false negatives on cross-lingual pairs than within-language ones at the same cutoff. Production systems serving MENA audiences would need language-aware threshold calibration, with separate decision boundaries for EN×EN, AR×AR, and EN×AR comparisons. Second, the domain overlap problem: the IPCC/COP28 cross-match scoring 0.7516 — higher than eight of ten same-topic pairs — demonstrates that within a specialized domain like climate policy, the model cannot reliably distinguish between two closely related topics. For applications where routing an article to the correct topic category matters, such as a multilingual content management system for a regional news outlet, the base multilingual model would require fine-tuning on labeled Arabic/English climate data to achieve acceptable precision. Third, the higher within-Arabic clustering observed here may reflect dataset characteristics specific to this corpus rather than a general property of Arabic climate text, and teams building production systems should validate these baselines on their own domain data before setting thresholds.
 
The practical recommendation is to treat `bert-base-multilingual-cased` as a strong general-purpose baseline for bilingual retrieval in the MENA region — it demonstrably works, requires no translation infrastructure, and serves 104 languages with a single model — but to invest in domain-specific fine-tuning and language-aware calibration before deploying it for classification or precision-sensitive ranking tasks. The cross-lingual capability is real; the question is whether 0.094 of separation between same-topic and off-topic pairs is enough for the specific precision requirements of the application at hand.
