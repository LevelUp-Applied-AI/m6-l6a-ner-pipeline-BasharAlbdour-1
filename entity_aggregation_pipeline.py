import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from collections import defaultdict
from itertools import combinations
from math import log
 
from ner_pipeline import extract_spacy_entities
 
 
NORMALIZATION_MAP = {
    # United Nations variants
    "un":                           "United Nations",
    "u.n.":                         "United Nations",
    "united nations":               "United Nations",
 
    # IPCC variants
    "ipcc":                         "IPCC",
    "intergovernmental panel on climate change": "IPCC",
 
    # USA variants
    "us":                           "United States",
    "u.s.":                         "United States",
    "united states":                "United States",
    "united states of america":     "United States",
    "america":                      "United States",
 
    # EU variants
    "eu":                           "European Union",
    "european union":               "European Union",
    "the european union":           "European Union",
 
    # UAE variants
    "uae":                          "United Arab Emirates",
    "united arab emirates":         "United Arab Emirates",
    "the united arab emirates":     "United Arab Emirates",
 
    # World Bank variants
    "world bank":                   "World Bank",
    "the world bank":               "World Bank",
 
    # UNFCCC variants
    "unfccc":                       "UNFCCC",
    "un climate body":              "UNFCCC",
 
    # COP variants
    "cop28":                        "COP28",
    "cop 28":                       "COP28",
    "cop27":                        "COP27",
    "cop 27":                       "COP27",
    "cop26":                        "COP26",
    "cop 26":                       "COP26",
 
    # Paris Agreement variants
    "paris agreement":              "Paris Agreement",
    "the paris agreement":          "Paris Agreement",
    "paris accord":                 "Paris Agreement",
    "paris climate agreement":      "Paris Agreement",
    "paris climate accord":         "Paris Agreement",
 
    # UNEP variants
    "unep":                         "UNEP",
    "un environment programme":     "UNEP",
 
    # Green Climate Fund variants
    "gcf":                          "Green Climate Fund",
    "green climate fund":           "Green Climate Fund",
    "the green climate fund":       "Green Climate Fund",
 
    # Middle East variants
    "the middle east":              "Middle East",
    "middle east":                  "Middle East",
 
    # Jordan variants
    "jordan":                       "Jordan",
    "jordanian":                    "Jordan",
    "the hashemite kingdom of jordan": "Jordan",
 
    # China variants
    "china":                        "China",
    "people's republic of china":   "China",
 
    # India variants
    "india":                        "India",
    "republic of india":            "India",
}
 
 
def normalize_entity(text):
    
    return NORMALIZATION_MAP.get(text.lower().strip(), text.strip())
 
 
def apply_normalization(entities_df):
    
    df = entities_df.copy()
    df["canonical"] = df["entity_text"].apply(normalize_entity)
    return df
 
 
NOISE_TOKENS = {
    # Temporal adverbs
    "annually", "annual", "recently", "today", "previously",
    "currently", "early", "late", "last", "next", "past",
    "future", "recent", "weekly", "monthly", "daily",
    # Ordinals
    "first", "second", "third", "fourth", "fifth",
    # Small cardinal numbers (not years or quantities)
    "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten",
    # Standalone digits that slipped through
    "3", "4", "5", "6", "7", "8", "9",
    "10", "12", "15", "20", "25", "30", "40", "50",
    "60", "70", "80", "90",
    # Vague time spans
    "50 years", "20-year", "30-year", "10-year", "5-year",
    "last year", "this year", "next year",
    # Population figures that are too generic
    "15,000", "5,000",
}
 
 
def filter_noise(entities_df):
    
    mask = ~entities_df["canonical"].str.lower().str.strip().isin(NOISE_TOKENS)
    removed = (~mask).sum()
    print(f"  Noise tokens removed: {removed}")
    return entities_df[mask].reset_index(drop=True)
 
 
 
def compute_cooccurrence(entities_df, use_canonical=True):
    
    col = "canonical" if use_canonical and "canonical" in entities_df.columns \
          else "entity_text"
 
    adjacency = defaultdict(lambda: defaultdict(int))
 
    for text_id, group in entities_df.groupby("text_id"):
        entities = group[col].unique().tolist()
        for a, b in combinations(sorted(entities), 2):
            if a != b:
                adjacency[a][b] += 1
                adjacency[b][a] += 1
 
    rows = []
    seen = set()
    for a, neighbors in adjacency.items():
        for b, count in neighbors.items():
            pair = tuple(sorted([a, b]))
            if pair not in seen:
                seen.add(pair)
                rows.append({"entity_a": pair[0], "entity_b": pair[1], "count": count})
 
    edge_list = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return edge_list, dict(adjacency)
 
 
 
def compute_entity_tfidf(entities_df, df, use_canonical=True):
    
    col = "canonical" if use_canonical and "canonical" in entities_df.columns \
          else "entity_text"
 
    id_to_cat = df.set_index("id")["category"].to_dict()
    ents = entities_df.copy()
    ents["category"] = ents["text_id"].map(id_to_cat)
 
    categories  = sorted(ents["category"].dropna().unique())
    n_categories = len(categories)
 
    cat_entity_counts = (
        ents.groupby(["category", col])
        .size()
        .reset_index(name="count")
    )
 
    cat_totals = ents.groupby("category").size().to_dict()
 
    entity_cat_presence = (
        cat_entity_counts.groupby(col)["category"]
        .nunique()
        .to_dict()
    )
 
    rows = []
    for _, row in cat_entity_counts.iterrows():
        entity   = row[col]
        category = row["category"]
        count    = row["count"]
 
        tf  = count / cat_totals[category]
        idf = log(n_categories / entity_cat_presence.get(entity, 1))
        rows.append({
            "entity":   entity,
            "category": category,
            "count":    count,
            "tf":       round(tf,   4),
            "idf":      round(idf,  4),
            "tfidf":    round(tf * idf, 6),
        })
 
    result = pd.DataFrame(rows).sort_values(
        ["category", "tfidf"], ascending=[True, False]
    ).reset_index(drop=True)
 
    return result
 
 
 
def plot_cooccurrence_network(edge_list, top_n=20, output_path="tier2_network.png"):
    
    top_edges = edge_list.head(top_n)
 
    G = nx.Graph()
    for _, row in top_edges.iterrows():
        G.add_edge(row["entity_a"], row["entity_b"], weight=row["count"])
 
    degrees    = dict(G.degree())
    node_sizes = [300 + degrees[n] * 200 for n in G.nodes()]
 
    weights    = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    edge_widths = [1 + 5 * (w / max_weight) for w in weights]
 
    max_degree = max(degrees.values()) if degrees else 1
    node_colors = [degrees[n] / max_degree for n in G.nodes()]
 
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")
 
    pos = nx.spring_layout(G, seed=42, k=2.5)
 
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        alpha=0.6,
        edge_color=[plt.cm.YlOrRd(w / max_weight) for w in weights],
    )
 
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        alpha=0.9,
    )
 
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8,
        font_color="white",
        font_weight="bold",
    )
 
    top_edge_labels = {
        (row["entity_a"], row["entity_b"]): str(row["count"])
        for _, row in top_edges.head(10).iterrows()
        if G.has_edge(row["entity_a"], row["entity_b"])
    }
    nx.draw_networkx_edge_labels(
        G, pos, top_edge_labels, ax=ax,
        font_size=7, font_color="#FFD700",
    )
 
    ax.set_title(
        f"Top {top_n} Entity Co-occurrences\n"
        "(node size = degree, edge thickness = co-occurrence count)",
        fontsize=13, fontweight="bold", color="white", pad=15
    )
    ax.axis("off")
 
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                norm=plt.Normalize(vmin=0, vmax=max_degree))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Node degree", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print(f"Network visualization saved to: {output_path}")
 
 
 
if __name__ == "__main__":
 
    nlp     = spacy.load("en_core_web_sm")
    df      = pd.read_csv("data/climate_articles.csv")
 
 
    print("=" * 65)
    print("EXTRACTING ENTITIES")
    print("=" * 65)
 
    raw_entities = extract_spacy_entities(df, nlp)
    print(f"Raw entities extracted: {len(raw_entities)}")
 
 
    print("\n" + "=" * 65)
    print("1. ENTITY NORMALIZATION")
    print("=" * 65)
 
    normalized = apply_normalization(raw_entities)
    normalized = filter_noise(normalized)
 
    changed = normalized[normalized["entity_text"] != normalized["canonical"]]
    print(f"\nEntities normalized: {len(changed)} surface forms mapped to canonical")
    print("\nSample normalizations:")
    sample = (
        changed[["entity_text", "canonical"]]
        .drop_duplicates()
        .head(20)
    )
    for _, row in sample.iterrows():
        print(f"  '{row['entity_text']}' → '{row['canonical']}'")
 
    print("\nTop 20 canonical entities by frequency:")
    top_canonical = (
        normalized.groupby("canonical")
        .size()
        .sort_values(ascending=False)
        .head(20)
    )
    for entity, count in top_canonical.items():
        print(f"  {entity:<35} {count:>4}")
 
 
    print("\n" + "=" * 65)
    print("2. ENTITY CO-OCCURRENCE")
    print("=" * 65)
 
    edge_list, adjacency = compute_cooccurrence(normalized, use_canonical=True)
 
    print(f"\nTotal unique entity pairs: {len(edge_list)}")
    print("\nTop 20 co-occurring entity pairs:")
    print(f"\n{'Entity A':<30} {'Entity B':<30} {'Count':>6}")
    print("-" * 68)
    for _, row in edge_list.head(20).iterrows():
        print(f"{row['entity_a']:<30} {row['entity_b']:<30} {row['count']:>6}")
 
    edge_list.to_csv("tier2_cooccurrence_edges.csv", index=False)
    print("\nEdge list saved to: tier2_cooccurrence_edges.csv")
 
    top_entities = list(
        normalized.groupby("canonical")
        .size()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    adj_matrix = pd.DataFrame(0, index=top_entities, columns=top_entities)
    for a in top_entities:
        for b in top_entities:
            if a != b and b in adjacency.get(a, {}):
                adj_matrix.loc[a, b] = adjacency[a][b]
 
    print("\nAdjacency matrix (top 15 entities):")
    print(adj_matrix.to_string())
 
 
    print("\n" + "=" * 65)
    print("3. TF-IDF ENTITY IMPORTANCE BY CATEGORY")
    print("=" * 65)
 
    tfidf_df = compute_entity_tfidf(normalized, df, use_canonical=True)
 
    print("\nTop 10 most distinctive entities per category:")
    for category in sorted(df["category"].unique()):
        cat_df = tfidf_df[tfidf_df["category"] == category].head(10)
        print(f"\n--- {category.upper()} ---")
        print(f"{'Entity':<35} {'Count':>6} {'TF':>8} {'IDF':>8} {'TF-IDF':>10}")
        print("-" * 70)
        for _, row in cat_df.iterrows():
            print(f"{row['entity']:<35} {row['count']:>6} "
                  f"{row['tf']:>8.4f} {row['idf']:>8.4f} {row['tfidf']:>10.6f}")
 
    tfidf_df.to_csv("tier2_tfidf.csv", index=False)
    print("\nFull TF-IDF table saved to: tier2_tfidf.csv")
 
 
    print("\n" + "=" * 65)
    print("4. NETWORK VISUALIZATION")
    print("=" * 65)
 
    plot_cooccurrence_network(
        edge_list,
        top_n=20,
        output_path="tier2_network.png"
    )
 
    print("\nDone. Output files:")
    print("  tier2_cooccurrence_edges.csv")
    print("  tier2_tfidf.csv")
    print("  tier2_network.png")