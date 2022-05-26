from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('data.csv')

# Plot raw data
plt.scatter(data['CorrectAnswer'],data['AnswerTime'])
plt.show()

df = pd.DataFrame(data, columns=['CorrectAnswer', 'AnswerTime'])
print(df.head(2))
print(df.info)
print(df.describe())

# Elbow method to find the appropriate number of clusters
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10)).fit(df.values)
visualizer.show()

# Silhouette method to find the appropriate number of clusters
range_n_clusters = list (range(3,6))
silhouette_scores = []
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters)
    preds = kmeans.fit_predict(df.values)
    score = silhouette_score(df, preds)
    silhouette_scores.append(score)
    print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))

    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(df.values)
    visualizer.show()


plt.plot(range_n_clusters, silhouette_scores)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette score")
plt.show()

# K-Means clustering

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)
predict = kmeans.fit_predict(df.values)

# # all labels in the dataset
# print(kmeans.labels_)

# Plot clusters with centroids
sns.scatterplot(data=df, x="CorrectAnswer", y="AnswerTime", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()

# Writing labels into dataset
df["cluster"] = predict
colors = sns.color_palette()[0:4]
df = df.sort_values("cluster")

# Defining rename schema
cnames = {"0": "Unsuccessful", "1": "Successful", "2": "Average"}
df["Degree"] = [cnames[str(i)] for i in df.cluster]

# Plot to verify order
sns.scatterplot(data=df, x="CorrectAnswer", y="AnswerTime", hue=df["Degree"])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],marker="X", c="r", s=80, label="centroids")
plt.show()

# Save dataframe as CSV
df.to_csv("LabeledData.csv")

# # predicted class for a given value
# predicted_class = kmeans.predict([[100, 100]])
# print("Predicted class: ",predicted_class)


