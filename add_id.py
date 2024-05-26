import pandas as pd

data = pd.read_csv('data.csv')
data['id'] = range(len(data))
data = data[['id', 'publish_date', 'headline_text']]
data.to_csv('data_with_id.csv', index=False)
