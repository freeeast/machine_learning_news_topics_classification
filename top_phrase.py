import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import os

output_dir = '/Users/arctic_zhou/Desktop/archive/Ngram Distribution'
os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv('data.csv')

data['publish_date'] = pd.to_datetime(data['publish_date'], format='%Y%m%d')

data['headline_text'] = data['headline_text'].str.lower().str.replace('[^\w\s]', '')

vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 2), max_features=1000)
X = vectorizer.fit_transform(data['headline_text'])
bigrams = vectorizer.get_feature_names_out()

bigrams_df = pd.DataFrame(X.toarray(), columns=bigrams)
bigrams_df['year'] = data['publish_date'].dt.year

excluded_bigrams = ['country hour', 'man charged', 'pleads guilty']
bigrams_df.drop(columns=excluded_bigrams, inplace=True, errors='ignore')

annual_bigram_counts = bigrams_df.groupby('year').sum()

top_bigrams = annual_bigram_counts.sum().nlargest(5).index  

plt.figure(figsize=(12, 8))
for bigram in top_bigrams:
    plt.plot(annual_bigram_counts.index, annual_bigram_counts[bigram], label=f"Bigram: {bigram}", marker='o')

plt.title('Annual Trend for Top 5 Bigrams')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "top_bigrams_trend.png"))  
plt.close()

print(f"Chart for top bigrams has been saved to {output_dir}.")
