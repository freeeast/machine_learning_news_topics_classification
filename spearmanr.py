import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('hotness_metrics1.csv')
df['max_hotness_date'] = pd.to_datetime(df['max_hotness_date'])
df.head()

corr_matrix, _ = spearmanr(df.drop(['keyword', 'max_hotness_date'], axis=1))
corr_df = pd.DataFrame(corr_matrix, columns=df.drop(['keyword', 'max_hotness_date'], axis=1).columns, index=df.drop(['keyword', 'max_hotness_date'], axis=1).columns)
print("Spearman Correlation Matrix:")
print(corr_df)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', annot_kws={"size": 10})
plt.title('Spearman Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.show()

data_norm = (df.drop(['keyword', 'max_hotness_date'], axis=1) - df.drop(['keyword', 'max_hotness_date'], axis=1).min()) / (df.drop(['keyword', 'max_hotness_date'], axis=1).max() - df.drop(['keyword', 'max_hotness_date'], axis=1).min())
entropy = -1 / np.log(len(df)) * np.sum(data_norm * np.log(data_norm + 1e-10), axis=0)
weights = (1 - entropy) / np.sum(1 - entropy)
print("Feature Weights:")
print(dict(zip(df.drop(['keyword', 'max_hotness_date'], axis=1).columns, weights)))
weights_df = pd.DataFrame({'Feature': df.drop(['keyword', 'max_hotness_date'], axis=1).columns, 'Weight': weights})
weights_df = weights_df.sort_values('Weight', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Weight', y='Feature', data=weights_df, palette='viridis')
plt.title('Feature Weights', fontsize=16)
plt.xlabel('Weight', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()