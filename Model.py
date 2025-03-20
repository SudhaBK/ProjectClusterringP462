#  we need to Import all the necessary libraries for data analysis
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

# Now loading our dataset 
df = pd.read_csv('World_development_mesurement.csv')

# Now, need to find columns with non-numerical data (like words or text)
categorical_cols = df.select_dtypes(include=['object']).columns

# Here we need to Convert non-numerical data into numbers
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Removing the symbols (percentage signs) and convert to decimal numbers
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = df[col].str.replace('%', '').astype('float') / 100
        except ValueError:
            pass

# Now, we need to fill the missing values with average numbers
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Create a clustering model with 5 groups (to group similar countries together)
kmeans = KMeans(n_clusters=5, n_init='auto')

# Now, we will Train the clustering model on our data so it can learn the patterns
kmeans.fit(df)

# Here we need to Save the trained model, in pickel file so that we can use it later.
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)