# Zomato_Restaurant_reviews | 
# Clustering and Sentiment Analysis of Zomato Restaurants
## Objective
- Group restaurants based on customer sentiments, pricing, and ratings using unsupervised learning.
- Extract business insights and help Zomato identify top-performing restaurants and improvement areas.
## Dataset Overview
- The dataset includes restaurant details, cost, customer reviews, and sentiments.
- Features: `Restaurant`, `avg_rating`, `review_count`, `Sentiment`, `Cost`, `positive_ratio`, `negative_ratio`, `neutral_ratio`, etc.
## Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram

# Load the data
df = pd.read_csv('zomato_reviews.csv')  # change to actual file

# Handle missing values
df.dropna(inplace=True)

# Convert numeric fields
df['Cost'] = df['Cost'].astype(float)
df['avg_rating'] = df['avg_rating'].astype(float)
df['review_count'] = df['review_count'].astype(int)
## Exploratory Data Analysis (EDA)
sns.countplot(data=df, x='Sentiment')
plt.title('Sentiment Distribution')
plt.show()
## Feature Engineering
# Group by restaurant and calculate sentiment ratios
grouped = df.groupby('Restaurant').agg({
    'avg_rating': 'mean',
    'review_count': 'sum',
    'Cost': 'mean',
    'Sentiment': lambda x: x.value_counts(normalize=True)
}).unstack().fillna(0)

# Flatten multiindex
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

# Rename sentiment ratio columns
grouped.rename(columns={
    'Sentiment_Negative': 'negative_ratio',
    'Sentiment_Neutral': 'neutral_ratio',
    'Sentiment_Positive': 'positive_ratio'
}, inplace=True)
## K-Means Clustering
# Select features for clustering
features = grouped[['avg_rating', 'review_count', 'Cost', 'positive_ratio', 'negative_ratio', 'neutral_ratio']]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
grouped['Cluster'] = kmeans.fit_predict(scaled_data)

# Analyze clusters
grouped.groupby('Cluster').mean()
## Hierarchical Clustering
# Linkage and dendrogram
Z = linkage(scaled_data, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=grouped.index.tolist(), leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Restaurant')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()
## Business Insights
- **Cluster 0**: High rating, positive reviews → Top-performing.
- **Cluster 1**: Mixed ratings → Medium-performing.
- **Cluster 2**: Low rating, more negative reviews → Underperforming.
## Recommendations
- Promote Cluster 0 for premium ads or Zomato Gold partnerships.
- For Cluster 1, suggest promotional campaigns or UI improvements.
- Cluster 2 restaurants need quality and service improvements.
- Use targeted marketing based on cluster performance
- Partner with high-performing restaurants for Zomato Gold and premium ads.
- Encourage poor performers to improve by highlighting key improvement areas (cost, reviews, etc.).
## Conclusion
- Successfully segmented restaurants using unsupervised learning.
- Provided actionable strategies for improving business outcomes based on clustering.
- Extracted valuable sentiment and performance-based groupings.
