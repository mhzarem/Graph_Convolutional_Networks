import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn import metrics
np.random.seed(1)

# Get your mentioned graph
G = nx.karate_club_graph()

# Get ground-truth: club-labels -> transform to 0/1 np-array
#     (possible overcomplicated networkx usage here)
gt_dict = nx.get_node_attributes(G, 'club')
print(gt_dict)
gt = [gt_dict[i] for i in G.nodes()]
gt = np.array([0 if i == 'Mr. Hi' else 1 for i in gt])

# Get adjacency-matrix as numpy-array
adj_mat = nx.to_numpy_matrix(G)
nx.draw_networkx(G)
plt.show()
print('ground truth')
print(gt)
color_or = []
for item in gt:
    if item == 1:
        color_or.append('red')
    else:
        color_or.append('green')


nx.draw_networkx(G, node_color=color_or)
plt.show()
# Cluster
sc = SpectralClustering(2, affinity='precomputed', n_init=100)
sc.fit(adj_mat)

# Compare ground-truth and clustering-results
print('spectral clustering')
print(sc.labels_)

print('just for better-visualization: invert clusters (permutation)')
print(np.abs(sc.labels_ - 1))
color_pre = []
for item in np.abs(sc.labels_ - 1):
    if item == 1:
        color_pre.append('red')
    else:
        color_pre.append('green')
nx.draw_networkx(G, node_color=color_pre)
plt.show()
print(gt)

# Calculate some clustering metrics
print(metrics.adjusted_rand_score(gt, sc.labels_))
print(metrics.confusion_matrix(gt, np.abs(sc.labels_ - 1)))
print(metrics.adjusted_mutual_info_score(gt, sc.labels_))