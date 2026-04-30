# Multilingual NER Comparison Table

**Label schema: native labels kept (Option A)**

| Language | Model | Total Entities | Entity Density | No-Entity Rate | Label Counts | Example Entities |
|---|---|---|---|---|---|---|
| English | spaCy xx | 93 | 7.62 per 100 words | 0/20 texts | ORG: 39, LOC: 27, MISC: 15, PER: 12 | IPCC (MISC), Sixth Assessment Report (MISC), Celsius (PER) |
| Arabic | spaCy xx | 20 | 2.15 per 100 words | 5/20 texts | PER: 9, MISC: 7, ORG: 2, LOC: 2 | وأكد التقرير (PER), وقّع الأردن (MISC), وأكد وزير (PER) |
| English | HF XLM-R | 93 | 7.62 per 100 words | 0/20 texts | ORG: 55, LOC: 28, PER: 10 | Antonio Guterres (PER), COP (ORG), Dubai (LOC) |
| Arabic | HF XLM-R | 64 | 6.88 per 100 words | 0/20 texts | ORG: 33, LOC: 29, PER: 2 | الهيئة الحكومية الدولية المعنية بتغير المناخ (ORG), الأردن (LOC), البنك الدولي (ORG) |