import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('imdb_top_1000.csv')

# Convert 'Released_Year' to numeric
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')

# Convert 'Runtime' to numeric by removing ' min'
df['Runtime'] = df['Runtime'].str.replace(' min', '')
df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')

# Convert 'IMDB_Rating' to numeric
df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')

# Convert 'Meta_score' to numeric
df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')

# Convert 'No_of_Votes' to numeric
df['No_of_Votes'] = df['No_of_Votes'].str.replace(',', '')
df['No_of_Votes'] = pd.to_numeric(df['No_of_Votes'], errors='coerce')

# Convert 'Gross' to numeric by removing ',' and '$'
df['Gross'] = df['Gross'].str.replace(',', '')
df['Gross'] = df['Gross'].str.replace('$', '')
df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')

# Handle missing values
df = df.dropna()

# Standardize numeric columns
scaler = StandardScaler()
df[['Released_Year', 'Runtime', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']] = scaler.fit_transform(df[['Released_Year', 'Runtime', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']])

# Save the cleaned and preprocessed dataset
df.to_csv('imdb_dataset_cleaned.csv', index=False)
