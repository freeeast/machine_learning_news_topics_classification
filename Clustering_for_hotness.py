import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

hotness_metrics_df = pd.read_csv('hotness_metrics1.csv', parse_dates=['max_hotness_date'], dtype={
    'max_hotness': float,
    'gt_mean_duration': float,
    'hotness_chg_rate': float,
    'hotness_area': float,
    'num_peaks': int,
    'avg_peak_interval': float,
    'rise_rate': float,
    'fall_rate': float,
    'std_dev': float,
    'autocorr': float,
    'seasonal_std':float
})

for i in range(5):
    hotness_metrics_df[f'num_peaks_{i}'] = hotness_metrics_df['num_peaks']
    hotness_metrics_df[f'seasonal_std_{i}'] = hotness_metrics_df['seasonal_std']

keywords1 = ['coronaviru', 'zealand', 'world', 'women', 'water']
keywords2 = ['anzac', 'christma']

hotness_metrics_df['is_1'] = hotness_metrics_df['keyword'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in keywords1) else 0)
hotness_metrics_df['is_2'] = hotness_metrics_df['keyword'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in keywords2) else 0)

scaler = StandardScaler()
scaled_metrics = scaler.fit_transform(hotness_metrics_df[['max_hotness', 'gt_mean_duration', 'hotness_chg_rate', 'hotness_area', 
                                                          'num_peaks', 'avg_peak_interval', 'rise_rate', 'fall_rate',
                                                          'std_dev', 'autocorr', 'seasonal_std', 'is_1', 'is_2']])

n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_metrics)

hotness_metrics_df['cluster'] = cluster_labels

print("Keywords:")
print(hotness_metrics_df['cluster'].value_counts())
print()

print("Average Hotness:")
print(hotness_metrics_df.groupby('cluster').mean(numeric_only=True))
print()

print("Representative keywords")
for cluster in range(n_clusters):
    cluster_keywords = hotness_metrics_df[hotness_metrics_df['cluster'] == cluster]['keyword']
    try:
        print(f"Cluster {cluster}: {', '.join(keyword[:20] for keyword in cluster_keywords.head(5))}")
    except TypeError:
        print(f"Clutser {cluster}: can not print")

hotness_metrics_df.to_csv('hotness_metrics_clustered1.csv', index=False)
print("Write in: hotness_metrics_clustered1.csv")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

cluster_counts = pd.Series([51, 329, 452, 155, 43], index=range(5))

plt.subplot(1, 2, 1)
sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
plt.xlabel("Cluster")
plt.ylabel("Number of Keywords")
plt.title("Number of Keywords per Cluster")

cluster_means = pd.DataFrame({
    'max_hotness': [0.8, 0.6, 0.9, 0.7, 0.5],
    'gt_mean_duration': [10, 20, 15, 25, 30],
    'hotness_chg_rate': [0.1, 0.2, 0.15, 0.25, 0.3],
    'hotness_area': [100, 200, 150, 250, 300],
    'num_peaks': [2, 4, 3, 5, 6],
    'avg_peak_interval': [5, 10, 7, 12, 15],
    'rise_rate': [0.05, 0.1, 0.08, 0.12, 0.15],
    'fall_rate': [0.03, 0.06, 0.04, 0.08, 0.1],
    'std_dev': [0.2, 0.4, 0.3, 0.5, 0.6],
    'autocorr': [0.7, 0.6, 0.8, 0.5, 0.4],
    'seasonal_std': [0.1, 0.2, 0.15, 0.25, 0.3]
}, index=range(5))

scaler = MinMaxScaler()
cluster_means_scaled = pd.DataFrame(scaler.fit_transform(cluster_means), columns=cluster_means.columns, index=cluster_means.index)

plt.subplot(1, 2, 2, polar=True)
angles = np.linspace(0, 2 * np.pi, len(cluster_means_scaled.columns), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

for i in range(len(cluster_means_scaled)):
    values = cluster_means_scaled.loc[i].values.tolist()
    values += values[:1]
    plt.polar(angles, values, linewidth=1, label=f"Cluster {i}")

plt.thetagrids(np.degrees(angles[:-1]), labels=cluster_means_scaled.columns, fontsize=8)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
plt.title("Mean Hotness Metrics per Cluster (Normalized)")

plt.tight_layout()
plt.show()