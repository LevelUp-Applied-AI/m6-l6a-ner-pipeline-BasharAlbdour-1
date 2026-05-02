import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
 
from ner_pipeline import extract_spacy_entities
 
 
 
def get_entity_counts_by_category(df, nlp):
    counts = defaultdict(lambda: defaultdict(int))
 
    english_df = df[df["language"] == "en"]
    for _, row in english_df.iterrows():
        doc = nlp(row["text"])
        for ent in doc.ents:
            counts[row["category"]][ent.label_] += 1
 
    return {cat: dict(label_counts) for cat, label_counts in counts.items()}
 
 
def evaluate_per_category(spacy_entities_df, gold_df, df):
    
    id_to_category = df.set_index("id")["category"].to_dict()
 
    pred_df = spacy_entities_df.copy()
    pred_df["category"] = pred_df["text_id"].map(id_to_category)
 
    gold_df_ = gold_df.copy()
    gold_df_["category"] = gold_df_["text_id"].map(id_to_category)
 
    results = {}
    for category in sorted(df["category"].unique()):
        pred_cat = pred_df[pred_df["category"] == category]
        gold_cat = gold_df_[gold_df_["category"] == category]
 
        if gold_cat.empty:
            results[category] = {
                "precision": None, "recall": None, "f1": None,
                "tp": 0, "fp": 0, "fn": 0,
                "note": "no gold annotations in this category"
            }
            continue
 
        pred_set = set(zip(
            pred_cat["text_id"],
            pred_cat["entity_text"].str.lower(),
            pred_cat["entity_label"],
        ))
        gold_set = set(zip(
            gold_cat["text_id"],
            gold_cat["entity_text"].str.lower(),
            gold_cat["entity_label"],
        ))
 
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
 
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
 
        results[category] = {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }
 
    return results
 
 
def build_counts_matrix(counts_by_category):
    
    all_labels = sorted({
        label
        for cat_counts in counts_by_category.values()
        for label in cat_counts
    })
    categories = sorted(counts_by_category.keys())
 
    data = {
        cat: [counts_by_category[cat].get(label, 0) for label in all_labels]
        for cat in categories
    }
    return pd.DataFrame(data, index=all_labels)
 
 
def plot_heatmap(matrix_df, output_path="tier1_heatmap.png"):
    
    fig, ax = plt.subplots(figsize=(10, 12))
 
    data = matrix_df.values.astype(float)
 
    row_max = data.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    normalized = data / row_max
 
    im = ax.imshow(normalized, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
 
    ax.set_xticks(range(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index, fontsize=10)
 
    for i in range(len(matrix_df.index)):
        for j in range(len(matrix_df.columns)):
            count = int(data[i, j])
            color = "white" if normalized[i, j] > 0.6 else "black"
            ax.text(j, i, str(count), ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")
 
    plt.colorbar(im, ax=ax, label="Relative frequency (row-normalized)")
    ax.set_title("Entity Type Distribution by Category\n(color = row-normalized, numbers = raw counts)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Entity Type", fontsize=12)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to: {output_path}")
 
 
def plot_grouped_bar(counts_by_category, top_n=8, output_path="tier1_bar_chart.png"):
    
    total_counts = defaultdict(int)
    for cat_counts in counts_by_category.values():
        for label, count in cat_counts.items():
            total_counts[label] += count
 
    top_labels = [label for label, _ in
                  sorted(total_counts.items(), key=lambda x: -x[1])[:top_n]]
 
    categories = sorted(counts_by_category.keys())
    n_cats     = len(categories)
    n_labels   = len(top_labels)
    x          = np.arange(n_labels)
    width      = 0.8 / n_cats
 
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
 
    fig, ax = plt.subplots(figsize=(14, 7))
 
    for i, (cat, color) in enumerate(zip(categories, colors)):
        counts = [counts_by_category[cat].get(label, 0) for label in top_labels]
        bars = ax.bar(x + i * width, counts, width,
                      label=cat.capitalize(), color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
 
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        str(int(h)), ha="center", va="bottom",
                        fontsize=8, color="black")
 
    ax.set_xticks(x + width * (n_cats - 1) / 2)
    ax.set_xticklabels(top_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Entity Count", fontsize=12)
    ax.set_xlabel("Entity Type", fontsize=12)
    ax.set_title(f"Top {top_n} Entity Types by Category",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grouped bar chart saved to: {output_path}")
 
 
 
if __name__ == "__main__":
 
    nlp      = spacy.load("en_core_web_sm")
    df       = pd.read_csv("data/climate_articles.csv")
    gold_df  = pd.read_csv("data/gold_entities.csv")
 
    english_df = df[df["language"] == "en"]
 
 
    print("=" * 65)
    print("ENTITY COUNTS BY CATEGORY")
    print("=" * 65)
 
    counts_by_category = get_entity_counts_by_category(df, nlp)
 
    for category in sorted(counts_by_category.keys()):
        print(f"\n--- {category.upper()} ---")
        cat_counts = counts_by_category[category]
        for label, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"  {label:<20} {count:>5}")
        print(f"  {'TOTAL':<20} {sum(cat_counts.values()):>5}")
 
 
    print("\n" + "=" * 65)
    print("GOLD ANNOTATION COVERAGE BY CATEGORY")
    print("=" * 65)
 
    gold_text_ids   = set(gold_df["text_id"].unique())
    id_to_category  = df.set_index("id")["category"].to_dict()
    gold_categories = [id_to_category[tid] for tid in gold_text_ids
                       if tid in id_to_category]
 
    coverage = pd.Series(gold_categories).value_counts()
    for cat, count in coverage.items():
        total = (df["category"] == cat).sum()
        print(f"  {cat:<15} {count} gold texts out of {total} total")
 
 
    print("\n" + "=" * 65)
    print("NER EVALUATION PER CATEGORY (standard labels, gold texts only)")
    print("=" * 65)
 
    spacy_entities = extract_spacy_entities(df, nlp)
    per_cat_metrics = evaluate_per_category(spacy_entities, gold_df, df)
 
    print(f"\n{'Category':<15} {'Precision':>10} {'Recall':>8} {'F1':>8} "
          f"{'TP':>5} {'FP':>5} {'FN':>5} {'Note'}")
    print("-" * 75)
    for cat, m in sorted(per_cat_metrics.items()):
        if m["precision"] is None:
            print(f"{cat:<15} {'—':>10} {'—':>8} {'—':>8} "
                  f"{'—':>5} {'—':>5} {'—':>5}  {m['note']}")
        else:
            print(f"{cat:<15} {m['precision']:>10} {m['recall']:>8} "
                  f"{m['f1']:>8} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")
 
 
    matrix_df = build_counts_matrix(counts_by_category)
 
    print("\n" + "=" * 65)
    print("ENTITY COUNT MATRIX (rows=labels, cols=categories)")
    print("=" * 65)
    print(matrix_df.to_string())
 

    print("\n" + "=" * 65)
    print("GENERATING VISUALIZATIONS")
    print("=" * 65)
 
    plot_heatmap(matrix_df,          output_path="tier1_heatmap.png")
    plot_grouped_bar(counts_by_category, top_n=8, output_path="tier1_bar_chart.png")
 
    print("\nSee tier1_analysis.md for written analysis.")