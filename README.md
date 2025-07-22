# Zomato_Restaurant_reviews | 
# Clustering and Sentiment Analysis of Zomato Restaurants
## Objective
- Group restaurants based on customer sentiments, pricing, and ratings using unsupervised learning.
- Extract business insights and help Zomato identify top-performing restaurants and improvement areas.
## Dataset Overview
- The dataset includes restaurant details, cost, customer reviews, and sentiments.
- Features: `Restaurant`, `avg_rating`, `review_count`, `Sentiment`, `Cost`, `positive_ratio`, `negative_ratio`, `neutral_ratio`, etc.
## Data Understanding & Preparation
We started by importing and exploring the dataset, which includes restaurant names, reviews, cost, and average ratings. The key challenge here was that we had unstructured textual data – customer reviews – which needed to be processed before any modeling.
We cleaned the data by removing nulls, converting relevant columns into numeric formats, and filtering out irrelevant or sparse rows. Then we moved on to analyzing the review text.
## Exploratory Data Analysis (EDA)
- Bar plots of sentiment vs cost revealed how customer perception varies with price.
- Correlation heatmaps helped identify the most influential variables for clustering.
- Countplot for simpler understanding
sns.countplot(data=df, x='Sentiment')
plt.title('Sentiment Distribution')
plt.show()
## Sentiment Analysis & Feature Engineering
To make our reviews more meaningful, we used NLP techniques to perform sentiment analysis. Here's how:
-Each review was passed through text preprocessing steps — lowercasing, removing punctuation, stop words, and lemmatization.
- We then calculated sentiment scores for each review — positive, negative, and neutral — using a sentiment lexicon.
- We created derived features: positive_ratio, negative_ratio, neutral_ratio, and also encoded overall Sentiment as Positive, Negative, or Neutral.
We also included features like:
- avg_rating — average customer rating.
- review_count — number of reviews.
- Cost — approximate cost for two people.
This transformed our dataset into a structured numerical form, ready for clustering."
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
- To group restaurants into similar segments, we applied KMeans clustering. Here's how:
- We scaled the numerical features using StandardScaler.
- Used the Elbow Method to determine the optimal number of clusters.
- Based on the elbow plot, we chose k=3 clusters.
After applying KMeans:
- We assigned each restaurant to a cluster using the grouped['Cluster'] column.
- We summarized each cluster's average rating, sentiment, and cost.
✅ For example, Cluster 0 had highly-rated and expensive restaurants, while Cluster 2 had more negative sentiments and lower ratings.
We visualized these clusters using scatter plots for easier interpretation.
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
- In addition to KMeans, we applied Hierarchical Clustering to see how restaurants relate to one another:
- We used the linkage method with 'ward' distance.
- The dendrogram helped visualize the tree of clusters.
- This was especially useful to see which restaurants are most similar and how clusters merge at different distance thresholds.
- This step added interpretability and confirmed the robustness of our earlier KMeans clusters.
# Linkage and dendrogram
Z = linkage(scaled_data, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=grouped.index.tolist(), leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Restaurant')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()
## Cluster comparison
We also created a confusion matrix-style table to compare KMeans clusters with Hierarchical clusters. This cross-tabulation gave insights into consistency between methods and helped validate our segmentation.
For example, most restaurants from Cluster 0 in KMeans were also grouped together in one dendrogram branch, confirming consistency.
## Business Insights
- **Cluster 0**: High rating, positive reviews → Top-performing.
- **Cluster 1**: Mixed ratings → Medium-performing.
- **Cluster 2**: Low rating, more negative reviews → Underperforming.
## Recommendations
1. Promote High-Rated, High-Cost Restaurants (Cluster 0): These are great for premium loyalty programs and influencer tie-ups.
2. Support Affordable Yet Popular Restaurants (Cluster 1): Promote these during peak seasons or through combo offers.
3. Rehabilitate Low-Rated Restaurants (Cluster 2): Target them with quality improvement feedback, training, or customer recovery campaigns.
4. Use targeted marketing based on cluster performance.
5. Partner with high-performing restaurants for Zomato Gold and premium ads.
6. Encourage poor performers to improve by highlighting key improvement areas (cost, reviews, etc.).
## Conclusion
To conclude, this project combined natural language processing, sentiment analysis, clustering, and business intelligence to create actionable insights from unstructured review data. By grouping restaurants based on customer sentiment, cost, and quality, we can personalize marketing, improve service quality, and drive business growth.

- Successfully segmented restaurants using unsupervised learning.
- Provided actionable strategies for improving business outcomes based on clustering.
- Extracted valuable sentiment and performance-based groupings.
