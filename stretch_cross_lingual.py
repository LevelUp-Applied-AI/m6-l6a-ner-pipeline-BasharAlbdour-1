import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "bert-base-multilingual-cased"
HEATMAP_PATH = "stretch_heatmap.png"

TOPICS = [
    {"label": "IPCC Assessment Report",     "en_kw": "IPCC",                          "ar_kw": "الهيئة الحكومية"},
    {"label": "COP28 Dubai",                "en_kw": "COP28",                         "ar_kw": "مؤتمر الأطراف الثامن"},
    {"label": "World Bank Climate Fund",    "en_kw": "World Bank",                    "ar_kw": "البنك الدولي"},
    {"label": "FAO Food Security",          "en_kw": "FAO",                           "ar_kw": "الأغذية والزراعة"},
    {"label": "NASA Hottest Year",          "en_kw": "NASA",                          "ar_kw": "ناسا"},
    {"label": "Greenland Ice Loss",         "en_kw": "Greenland",                     "ar_kw": "غرينلاند"},
    {"label": "WMO Sea Level Rise",         "en_kw": "World Meteorological",          "ar_kw": "المنظمة العالمية للأرصاد"},
    {"label": "Jordan Water Scarcity",      "en_kw": "water-scarce",                  "ar_kw": "شحاً في المياه"},
    {"label": "Amman Heatwave",             "en_kw": "Amman experienced its hottest", "ar_kw": "عمّان صيفها الأكثر"},
    {"label": "CO2 Emissions 2023",         "en_kw": "36.8 billion tonnes",           "ar_kw": "36.8 مليار طن"},
]

def select_paired_texts(filepath="data/climate_articles.csv"):
    
    df=pd.read_csv(filepath)
    en_df=df[df["language"]=="en"]
    ar_df=df[df["language"]=="ar"]
    
    pairs=[]
    for topic in TOPICS:
        en_match = en_df[en_df["text"].str.contains(topic["en_kw"], case=False, na=False)]
        ar_match = ar_df[ar_df["text"].str.contains(topic["ar_kw"], na=False)]
        
        if len(en_match) == 0 or len(ar_match) == 0:
            print(f"  [MISS] {topic['label']}")
            continue
 
        pairs.append({
            "label":   topic["label"],
            "en_id":   en_match.iloc[0]["id"],
            "en_text": en_match.iloc[0]["text"],
            "ar_id":   ar_match.iloc[0]["id"],
            "ar_text": ar_match.iloc[0]["text"],
        })
 
    return pairs

def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

def extract_embeddings(texts, tokenizer, model):
    embeddings = []
    for i, text in enumerate(texts):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            output = model(**encoded)
        emb = mean_pool(output, encoded["attention_mask"])
        embeddings.append(emb.squeeze().numpy())
        print(f"  [{i + 1}/{len(texts)}] done")
    return np.array(embeddings)

def build_similarity_matrix(en_embeddings, ar_embeddings):
    return cosine_similarity(en_embeddings, ar_embeddings)

def plot_heatmap(sim_matrix, pairs, save_path=HEATMAP_PATH):
    en_labels = [f"EN-{p['en_id']}: {p['en_text'][:40]}…" for p in pairs]
    ar_labels = [f"AR-{p['ar_id']}: {p['ar_text'][:35]}…" for p in pairs]
 
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(
        sim_matrix,
        xticklabels=ar_labels,
        yticklabels=en_labels,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.55,
        vmax=0.80,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "Cross-Lingual Cosine Similarity — bert-base-multilingual-cased\n"
        "English (rows) × Arabic (columns) | Climate Articles Dataset",
        fontsize=13,
        pad=14,
    )
    ax.set_xlabel("Arabic texts", fontsize=11)
    ax.set_ylabel("English texts", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap saved → {save_path}")
    
def print_analysis(sim_matrix, pairs, en_embeddings, ar_embeddings):
    
    n = len(pairs)
 
    print("\n── Same-topic pairs (diagonal) ─────────────────────────────────")
    diagonal = [sim_matrix[i, i] for i in range(n)]
    for i, score in enumerate(diagonal):
        p = pairs[i]
        print(f"  {p['label']:<30} EN-{p['en_id']} × AR-{p['ar_id']} : {score:.4f}")
 
    off_diag = [sim_matrix[i, j] for i in range(n) for j in range(n) if i != j]
    print(f"\n  Mean same-topic similarity  : {np.mean(diagonal):.4f}")
    print(f"  Mean off-diagonal similarity: {np.mean(off_diag):.4f}")
    print(f"  Gap                         : {np.mean(diagonal) - np.mean(off_diag):.4f}")
 
    print("\n── Top-3 most similar cross-lingual pairs ──────────────────────")
    flat = [(sim_matrix[i, j], i, j) for i in range(n) for j in range(n)]
    for score, i, j in sorted(flat, reverse=True)[:3]:
        print(f"  EN-{pairs[i]['en_id']} ({pairs[i]['label']}) × AR-{pairs[j]['ar_id']} ({pairs[j]['label']}) : {score:.4f}")

    print(f"\n── Within-language baselines ───────────────────────────────────")
    en_sim = cosine_similarity(en_embeddings)
    within_en = [en_sim[i, j] for i in range(n) for j in range(n) if i != j]
    print(f"  Mean within-EN similarity   : {np.mean(within_en):.4f}")

    ar_sim = cosine_similarity(ar_embeddings)
    within_ar = [ar_sim[i, j] for i in range(n) for j in range(n) if i != j]
    print(f"  Mean within-AR similarity   : {np.mean(within_ar):.4f}")
 
 
if __name__ == "__main__":
    print("Loading data and selecting paired texts...")
    pairs = select_paired_texts()
    for p in pairs:
        print(f"  [{p['label']}]  EN-{p['en_id']} × AR-{p['ar_id']}")
    print(f"Total pairs: {len(pairs)}\n")
 
    print(f"Loading model: {MODEL_NAME}  (~680MB, downloads on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.\n")
 
    print("Extracting English embeddings...")
    en_embeddings = extract_embeddings([p["en_text"] for p in pairs], tokenizer, model)
 
    print("\nExtracting Arabic embeddings...")
    ar_embeddings = extract_embeddings([p["ar_text"] for p in pairs], tokenizer, model)
 
    print("\nComputing cosine similarity matrix...")
    sim_matrix = build_similarity_matrix(en_embeddings, ar_embeddings)
    print(f"Matrix shape: {sim_matrix.shape}")
 
    print_analysis(sim_matrix, pairs, en_embeddings, ar_embeddings)
 
    plot_heatmap(sim_matrix, pairs)
    print("\nDone.")