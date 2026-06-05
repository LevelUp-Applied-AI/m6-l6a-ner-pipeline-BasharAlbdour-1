import pandas as pd
from transformers import pipeline
import spacy


SAMPLE_SIZE = 20          
HF_MODEL    = "Davlan/xlm-roberta-base-wikiann-ner"
SPACY_MODEL = "xx_ent_wiki_sm"


def load_data(filepath="data/climate_articles.csv"):
    df = pd.read_csv(filepath)
    english = df[df['language'] == 'en'].head(SAMPLE_SIZE).reset_index(drop=True)
    arabic  = df[df['language'] == 'ar'].head(SAMPLE_SIZE).reset_index(drop=True)
    print(f"Loaded {len(english)} English texts and {len(arabic)} Arabic texts")
    return english, arabic


def load_models():
    print(f"\nLoading spaCy model: {SPACY_MODEL}")
    nlp = spacy.load(SPACY_MODEL)
    print(f"Loading HuggingFace model: {HF_MODEL}")
    hf_ner = pipeline("ner", model=HF_MODEL, aggregation_strategy="simple")
    print("Models loaded successfully")
    return nlp, hf_ner

def run_spacy_ner(df, nlp):
    
    rows = []
    no_entity_count = 0

    for _, row in df.iterrows():
        doc = nlp(row['text'])
        if len(doc.ents) == 0:
            no_entity_count += 1
        for ent in doc.ents:
            rows.append({
                'text_id':      row['id'],
                'text':         row['text'][:60],
                'entity_text':  ent.text,
                'entity_label': ent.label_,
                'model':        'spacy_xx'
            })

    result = pd.DataFrame(rows, columns=['text_id', 'text', 'entity_text', 'entity_label', 'model'])
    return result, no_entity_count


def run_hf_ner(df, hf_ner):
    
    rows = []
    no_entity_count = 0

    for _, row in df.iterrows():
        results = hf_ner(row['text'])
        if len(results) == 0:
            no_entity_count += 1
        for ent in results:
            rows.append({
                'text_id':      row['id'],
                'text':         row['text'][:60],
                'entity_text':  ent['word'],
                'entity_label': ent['entity_group'],
                'model':        'hf_xlmr'
            })

    result = pd.DataFrame(rows, columns=['text_id', 'text', 'entity_text', 'entity_label', 'model'])
    return result, no_entity_count

def build_comparison_table(spacy_en, spacy_ar, hf_en, hf_ar,
                            spacy_en_no_ent, spacy_ar_no_ent,
                            hf_en_no_ent, hf_ar_no_ent,
                            english_df, arabic_df):

    combinations = [
        ('English', 'spaCy xx',  spacy_en,  spacy_en_no_ent, english_df),
        ('Arabic',  'spaCy xx',  spacy_ar,  spacy_ar_no_ent, arabic_df),
        ('English', 'HF XLM-R', hf_en,     hf_en_no_ent,    english_df),
        ('Arabic',  'HF XLM-R', hf_ar,     hf_ar_no_ent,    arabic_df),
    ]

    summary_rows = []
    detail_rows  = []

    for lang, model, df_ents, no_ent_count, source_df in combinations:

        total_entities   = len(df_ents)
        total_texts      = len(source_df)
        total_words      = source_df['text'].str.split().str.len().sum()
        entity_density   = round((total_entities / total_words) * 100, 2)
        no_entity_rate   = f"{no_ent_count}/{total_texts}"

        if not df_ents.empty:
            type_counts = df_ents['entity_label'].value_counts().to_dict()
        else:
            type_counts = {}

        if not df_ents.empty:
            examples = (
                df_ents.drop_duplicates(subset=['entity_text', 'entity_label'])
                .head(3)[['entity_text', 'entity_label']]
                .apply(lambda r: f"{r['entity_text']} ({r['entity_label']})", axis=1)
                .tolist()
            )
        else:
            examples = []

        summary_rows.append({
            'Language':          lang,
            'Model':             model,
            'Total Entities':    total_entities,
            'Entity Density':    f"{entity_density} per 100 words",
            'No Entity Rate':    no_entity_rate,
            'Label Counts':      type_counts,
            'Example Entities':  examples,
        })

    return summary_rows


def print_comparison_table(summary_rows):
    """Print the comparison table in a readable format."""
    print(f"\n{'='*70}")
    print("MULTILINGUAL NER COMPARISON TABLE")
    print(f"Label schema: native labels kept (Option A)")
    print(f"{'='*70}")

    for row in summary_rows:
        print(f"\nLanguage : {row['Language']}")
        print(f"Model    : {row['Model']}")
        print(f"{'─'*40}")
        print(f"Total entities found : {row['Total Entities']}")
        print(f"Entity density       : {row['Entity Density']}")
        print(f"No-entity rate       : {row['No Entity Rate']} texts")
        print(f"Label counts         : {row['Label Counts']}")
        print(f"Example entities     :")
        for ex in row['Example Entities']:
            print(f"  - {ex}")
        print(f"{'─'*40}")

    print(f"\n{'='*70}")
    
def save_comparison_table(summary_rows, output_path="stretch_comparison_table.md"):
    """Save the comparison table as a markdown file."""
    lines = []
    lines.append("# Multilingual NER Comparison Table\n")
    lines.append("**Label schema: native labels kept (Option A)**\n")
    lines.append("| Language | Model | Total Entities | Entity Density | No-Entity Rate | Label Counts | Example Entities |")
    lines.append("|---|---|---|---|---|---|---|")

    for row in summary_rows:
        examples = ", ".join(row['Example Entities'])
        label_counts = ", ".join(f"{k}: {v}" for k, v in row['Label Counts'].items())
        lines.append(
            f"| {row['Language']} "
            f"| {row['Model']} "
            f"| {row['Total Entities']} "
            f"| {row['Entity Density']} "
            f"| {row['No Entity Rate']} texts "
            f"| {label_counts} "
            f"| {examples} |"
        )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\nComparison table saved to {output_path}")

if __name__ == "__main__":
    english_df, arabic_df = load_data()
    nlp, hf_ner = load_models()

    print("\n--- Running spaCy on English ---")
    spacy_en, spacy_en_no_ent = run_spacy_ner(english_df, nlp)
    print(f"Entities found: {len(spacy_en)} | Texts with no entities: {spacy_en_no_ent}")

    print("\n--- Running spaCy on Arabic ---")
    spacy_ar, spacy_ar_no_ent = run_spacy_ner(arabic_df, nlp)
    print(f"Entities found: {len(spacy_ar)} | Texts with no entities: {spacy_ar_no_ent}")

    print("\n--- Running HF on English ---")
    hf_en, hf_en_no_ent = run_hf_ner(english_df, hf_ner)
    print(f"Entities found: {len(hf_en)} | Texts with no entities: {hf_en_no_ent}")

    print("\n--- Running HF on Arabic ---")
    hf_ar, hf_ar_no_ent = run_hf_ner(arabic_df, hf_ner)
    print(f"Entities found: {len(hf_ar)} | Texts with no entities: {hf_ar_no_ent}")
    
    summary = build_comparison_table(
        spacy_en, spacy_ar, hf_en, hf_ar,
        spacy_en_no_ent, spacy_ar_no_ent,
        hf_en_no_ent, hf_ar_no_ent,
        english_df, arabic_df
    )
    print_comparison_table(summary)
    save_comparison_table(summary)