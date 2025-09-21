import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt

# Keyword extractor using TF-IDF
def top_keywords_region(texts, topn=10):
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    top_idx = scores.argsort()[::-1][:topn]
    return [vec.get_feature_names_out()[i] for i in top_idx]

# Region based analysis
def region_keywords(df_region, text_column="Narrative", topn=10):
    return top_keywords_region(df_region[text_column].tolist(), topn=topn)

# Neighborhood based analysis
def build_neighbors(df, dims=["UMAP1","UMAP2","UMAP3"], n_neighbors=30):
    coords = df[dims].values
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    return nn, coords

def local_keywords(idx, df, nn, coords, text_column="Narrative", topn=10):
    dists, neighbors = nn.kneighbors([coords[idx]])
    texts = df.iloc[neighbors[0]][text_column].tolist()
    return top_keywords_region(texts, topn)

def analyze_keywords(df, label="UMAP Analysis", text_column="Narrative"):
    nn, coords = build_neighbors(df)
    mid_idx = df.index[len(df) // 2]

    print(f"\n {label} - Local Neighborhood Keywords:")
    local_top = local_keywords(mid_idx, df, nn, coords, text_column=text_column, topn=15)
    print(local_top)

    print(f"\n {label} - Region-Based Keywords:")
    region_top = region_keywords(df[df["UMAP1"] > 0], text_column=text_column, topn=15)
    print(region_top)

    return local_top, region_top

def extract_keywords_all_labels(df, text_lookup, label_column, top_k=15, include_clusters=True):
    results = []
    if include_clusters and "Cluster" in df.columns:
        grouped = df.groupby([label_column, "Cluster"])
    else:
        grouped = df.groupby(label_column)

    for keys, subset in grouped:
        if isinstance(keys, tuple):
            label_value, cluster_id = keys
        else:
            label_value, cluster_id = keys, None

        texts = [text_lookup[f] for f in subset['Filename'] if f in text_lookup]
        if not texts:
            continue

        vec = TfidfVectorizer(max_features=top_k, stop_words='english')
        X = vec.fit_transform(texts)
        keywords = vec.get_feature_names_out()

        results.append({
            "LabelColumn": label_column,
            "LabelValue": label_value,
            "Cluster": cluster_id,
            "Keywords": ", ".join(keywords)
        })

    return pd.DataFrame(results)

def neat_print_keywords(df_keywords, title, top_k=10):
    print(f"\n----{title} ----")
    df_sorted = df_keywords.sort_values(by=["Cluster", "LabelValue"], na_position="last")
    last_cluster = None
    for _, row in df_sorted.iterrows():
        cluster = row["Cluster"]
        label_value = row["LabelValue"]
        keywords = row["Keywords"].split(", ")[:top_k]
        if cluster != last_cluster:
            print(f"\n----Cluster {cluster} ----")
            last_cluster = cluster
        print(f"[{label_value}]: {', '.join(keywords)}")

def cluster_umap(df, dims=["UMAP1", "UMAP2", "UMAP3"], min_cluster_size=15):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    df["Cluster"] = clusterer.fit_predict(df[dims].values)
    return df

def cluster_sizes(df, cluster_col="Cluster"):
    counts = df[cluster_col].value_counts().sort_index()
    print("\nCluster sizes:")
    for cid, count in counts.items():
        print(f"  Cluster {cid}: {count} reports")
    return counts

def extract_hdbscan_keywords(df, text_lookup, cluster_col="Cluster", filename_col="Filename", top_k=15):
    rows = []
    for cluster_id in sorted(df[cluster_col].unique()):
        subset = df[df[cluster_col] == cluster_id]
        filenames = subset[filename_col].tolist()
        texts = [text_lookup.get(fn, "") for fn in filenames]
        texts = [t for t in texts if t and len(t.split()) > 5]
        if not texts:
            continue
        vec = TfidfVectorizer(max_features=top_k, stop_words="english")
        X = vec.fit_transform(texts)
        keywords = vec.get_feature_names_out().tolist()
        rows.append({"Cluster": cluster_id, "Count": len(texts), "Keywords": keywords})
    return pd.DataFrame(rows).sort_values("Cluster")

def neat_print_hdbscan_keywords(df_kws, title):
    print(f"\n=== {title} ===")
    for _, r in df_kws.iterrows():
        cluster = r["Cluster"]
        cnt = r["Count"]
        kw = ", ".join(r["Keywords"][:12])
        print(f"Cluster {cluster} ({cnt} reports): {kw}")

def plot_topN_heatmap(ct, top_n_rows=10, top_n_cols=None, title="", cmap="Blues", col_map=None):
    """
    Plot a heatmap for the top N rows and optionally top N columns of a crosstab.
    Sorts rows and columns by total frequency.

    Parameters:
    - ct: crosstab dataframe
    - top_n_rows: number of top rows to keep
    - top_n_cols: number of top columns to keep (optional)
    - title: plot title
    - cmap: heatmap color map
    - col_map: optional dict to map column values to labels
    """
    # Pick top N rows
    top_rows = ct.sum(axis=1).sort_values(ascending=False).head(top_n_rows).index
    ct_top = ct.loc[top_rows]

    # Pick top N columns if specified
    if top_n_cols is not None:
        top_cols = ct_top.sum(axis=0).sort_values(ascending=False).head(top_n_cols).index
        ct_top = ct_top[top_cols]

    # Map column values if provided
    if col_map is not None:
        ct_top.columns = [col_map.get(col, col) for col in ct_top.columns]

    # Plot
    plt.figure(figsize=(12, 6))
    sns.heatmap(ct_top, annot=True, fmt="d", cmap=cmap, cbar=True)
    plt.title(title)
    plt.ylabel(ct.index.name)
    plt.xlabel(ct.columns.name)
    plt.tight_layout()
    plt.show()

def run_hdbscan_assign(df, embedding, cluster_col="Cluster",
                       min_cluster_size=10, min_samples=5, metric='euclidean'):
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                          min_samples=min_samples,
                          metric=metric)
    labels = hdb.fit_predict(embedding)
    df = df.copy()
    df[cluster_col] = labels
    return df, hdb, labels
