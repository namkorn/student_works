import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift , estimate_bandwidth

X = np.loadtxt('data_clustering.txt', delimiter=',')

#прочитування data_clustering.txt і вивовидить вихідні точки на площину
plt.figure()
plt.scatter(X[:, 0], X[:, 1], color='black', s=80, marker='o', facecolors='none')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

#meanshift part - центри кластерів
bandwidth_X = estimate_bandwidth(X , quantile=0.15, n_samples=len(X))

meanshift_model = MeanShift(bandwidth= bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

cluster_centers = meanshift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =", num_clusters)

plt.figure()
markers = 'o*xvsd<'
for i, marker in zip(range(num_clusters), markers):
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='black')
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o', markerfacecolor= 'red', markeredgecolor= 'black', markersize= 15)

plt.title('Clusters')

# Ініціалізаця кластеризованих даних діаграмою
scores = []
values = np.arange(2, 10)

for num_clusters in values:
    # навчання ШІ на моделях
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)
    score = metrics.silhouette_score(X, kmeans.labels_,
                                     metric='euclidean', sample_size=len(X))

    print("\nNumber of clusters =", num_clusters)
    print("Silhouette score =", score)

    scores.append(score)

# вивід silhouette scores
plt.figure()
plt.bar(values, scores, width=0.5, color='black', align='center')
plt.title('Silhouette score vs number of clusters')

# показ кращих результатів
#num_clusters = np.argmax(scores) + values[0]
lables = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print('\nOptimal number of clusters =', num_clusters)


#kmeans part - кластеризовані дані з областями кластеризації


kmeans = KMeans(init= 'k-means++', n_clusters = num_clusters, n_init=10)
#навчання КMeans кластиризації на моделях
kmeans.fit(X)
#Розмір кроку сітки
step_size = 0.1
#Визначення сіток точок для нанесення меж
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
#Передбачення вихідних міток для всіх точок сітки
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

#Креслення різних регіонів та розфарбування їх
output = output.reshape(x_vals.shape)
plt.figure()
plt.clf()
plt.imshow(output, interpolation='nearest',
           extent=(x_vals.min(), x_vals.max(),
               y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired,
           aspect='auto',
           origin='lower')
# Накладання точок введення
plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none',
        edgecolors='black', s=80)
# Побудування центрів кластерів
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:,0], cluster_centers[:,1],
        marker='o', s=210, linewidths=4, color='black',
        zorder=12, facecolors='black')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Boundaries of clusters')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()
