# SCT_ML_2 – Customer Segmentation using K-Means Clustering

## Task
Create a K-Means clustering algorithm to group customers of a retail store based on their purchase history, using the Mall Customer Dataset.

## Approach
- Used Annual Income and Spending Score as clustering features
- Standardized features using StandardScaler
- Used the Elbow Method and Silhouette Score to determine optimal k
- Final model: K-Means with k=5 clusters (silhouette score ≈ 0.55)

## Results
Customers were grouped into 5 segments:
- High Income, High Spending (Target)
- High Income, Low Spending (Cautious)
- Low Income, High Spending (Impulsive)
- Low Income, Low Spending (Budget)
- Mid Income, Mid Spending (Standard)

## Files
- `task02_kmeans.py` – full clustering script
- `Mall_Customers.csv` – dataset
- `02_final_clusters.png` – final cluster visualization
- `01_elbow_silhouette.png` – k selection charts
- `03_cluster_demographics.png` – age/gender breakdown by cluster
- `Mall_Customers_Clustered.csv` – dataset with assigned cluster labels

## Tech Stack
Python, pandas, scikit-learn, matplotlib, seaborn
