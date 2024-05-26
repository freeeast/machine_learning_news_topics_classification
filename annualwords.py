import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import os

output_dir = '/Users/arctic_zhou/Desktop/archive/Yearly keywords'
os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv('data.csv')

data['publish_date'] = pd.to_datetime(data['publish_date'], format='%Y%m%d')

data['headline_text'] = data['headline_text'].str.lower().str.replace('[^\w\s]', '')
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(data['headline_text'])
keywords = vectorizer.get_feature_names_out()

keywords_df = pd.DataFrame(X.toarray(), columns=keywords)
keywords_df['year'] = data['publish_date'].dt.year

annual_keyword_counts = keywords_df.groupby('year').sum()

excluded_keywords = ['police', 'says', 'new', 'new','interview','man','says','govt','council','plan']
annual_keyword_counts.drop(columns=excluded_keywords, inplace=True, errors='ignore')

annual_top_keywords = annual_keyword_counts.idxmax(axis=1)
annual_top_frequencies = annual_keyword_counts.max(axis=1)

for year, keyword in annual_top_keywords.items():
    plt.figure(figsize=(10, 6))
    plt.plot(annual_keyword_counts.index, annual_keyword_counts[keyword], label=f"Keyword: {keyword}", marker='o')
    plt.title(f'Annual Trend for "{keyword}"')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{year}_{keyword}.png"))
    plt.close()

print(f"Charts have been saved to {output_dir}.")
