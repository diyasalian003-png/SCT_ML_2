"""
SkillCraft Technology - ML Internship
Task 02: K-Means Clustering on Mall Customer Dataset
Group retail customers based on Annual Income and Spending Score.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv("Mall_Customers.csv")
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# ----------------------------------------------------------------------
# 2. Feature selection
#    Classic version of this task clusters on Annual Income vs Spending
#    Score, which gives clean, interpretable 2D segments.
# ----------------------------------------------------------------------
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------
# 3. Elbow method + Silhouette score to choose k
# ----------------------------------------------------------------------
inertias = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(list(K_range), inertias, marker="o", color="#2E86AB")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")

axes[1].plot(list(K_range), sil_scores, marker="o", color="#E07A5F")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")

plt.tight_layout()
plt.savefig("01_elbow_silhouette.png", bbox_inches="tight")
plt.close()

best_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"\nBest k by silhouette score: {best_k}")
print("Silhouette scores:", dict(zip(K_range, [round(s, 3) for s in sil_scores])))

# For this dataset the well-known optimal segmentation is k=5
FINAL_K = 5

# ----------------------------------------------------------------------
# 4. Fit final model
# ----------------------------------------------------------------------
kmeans = KMeans(n_clusters=FINAL_K, init="k-means++", random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

final_sil = silhouette_score(X_scaled, df["Cluster"])
print(f"\nFinal model: k={FINAL_K}, silhouette score={final_sil:.3f}")

# Cluster centers back in original units
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(
    centers_original, columns=["Annual Income (k$)", "Spending Score (1-100)"]
)
centers_df.index.name = "Cluster"
print("\nCluster centers (original units):")
print(centers_df.round(1))

# ----------------------------------------------------------------------
# 5. Cluster profiling
# ----------------------------------------------------------------------
profile = df.groupby("Cluster").agg(
    Count=("CustomerID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean"),
).round(1)
print("\nCluster profile:")
print(profile)

# Human-readable labels based on income/spending combination
def label_cluster(row):
    income, spending = row["Avg_Income"], row["Avg_Spending"]
    if income >= 70 and spending >= 60:
        return "High Income, High Spending (Target)"
    if income >= 70 and spending < 40:
        return "High Income, Low Spending (Cautious)"
    if income < 40 and spending >= 60:
        return "Low Income, High Spending (Impulsive)"
    if income < 40 and spending < 40:
        return "Low Income, Low Spending (Budget)"
    return "Mid Income, Mid Spending (Standard)"

profile["Segment"] = profile.apply(label_cluster, axis=1)
print("\nSegment labels:")
print(profile[["Count", "Segment"]])

# ----------------------------------------------------------------------
# 6. Visualization: final clusters
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 6))
palette = sns.color_palette("Set1", FINAL_K)

for c in range(FINAL_K):
    subset = df[df["Cluster"] == c]
    plt.scatter(
        subset["Annual Income (k$)"],
        subset["Spending Score (1-100)"],
        s=60,
        color=palette[c],
        label=f"Cluster {c}: {profile.loc[c, 'Segment']}",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

plt.scatter(
    centers_df["Annual Income (k$)"],
    centers_df["Spending Score (1-100)"],
    s=250,
    color="black",
    marker="X",
    label="Centroids",
)

plt.title(f"Customer Segments via K-Means (k={FINAL_K})", fontsize=13)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=True)
plt.tight_layout()
plt.savefig("02_final_clusters.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 7. Extra: Gender / Age breakdown per cluster (bonus insight)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.countplot(data=df, x="Cluster", hue="Gender", ax=axes[0], palette="pastel")
axes[0].set_title("Gender distribution per cluster")

sns.boxplot(data=df, x="Cluster", y="Age", ax=axes[1], palette="pastel")
axes[1].set_title("Age distribution per cluster")

plt.tight_layout()
plt.savefig("03_cluster_demographics.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 8. Save labeled dataset
# ----------------------------------------------------------------------
out_df = df.copy()
out_df["Segment"] = out_df["Cluster"].map(profile["Segment"])
out_df.to_csv("Mall_Customers_Clustered.csv", index=False)

print("\nDone. Outputs saved to ")
