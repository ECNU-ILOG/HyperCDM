import numpy as np

from scipy.spatial.distance import cosine
from tqdm import tqdm


def cosine_similarity(X):
    n = X.shape[0]
    cos = np.zeros((n, n))
    for i in tqdm(range(n), "calculating similarity"):
        for j in range(i + 1, n):
            cos[i, j] = cos[j, i] = 1 - cosine(X[i], X[j])
    return cos


def euclidean_similarity(X):
    n = X.shape[0]
    dist = np.zeros((n, n))
    for i in tqdm(range(n), "calculating similarity"):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = np.linalg.norm(X[i] - X[j], 2)
    return dist

