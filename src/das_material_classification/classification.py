import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from tslearn.metrics import dtw


def load_data(path_dir):
    encoder = {"asphalt": 0, "sand": 1, "technical_chamber": 2}
    data_config_used = ["7", "12", "10"]
    dataset = []
    for config in data_config_used:
        for file in Path(path_dir).glob(f"DAS_Config{config}_*.npz"):
            dataset.append(np.load(file))
    n = len(dataset)
    series_length = len(dataset[0]['time_series'])
    X, y = np.zeros((n, series_length)), np.zeros(n, dtype=int)
    for i, data in enumerate(dataset):
        X[i] = data["time_series"]
        y[i] = encoder[data["material"].item()]
    return X, y


def create_distance_matrix(X, distance):
    n = len(X)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        distance_matrix[i, i] = 0
        for j in range(i + 1, n):
            dist = distance(X[i], X[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix


def scores_of_classification(distance_matrix, y, k=1):
    classifier = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
    loo = LeaveOneOut()
    scores = cross_val_score(classifier, distance_matrix, y, cv=loo)
    return scores

def euclidean_distance(x,y):
  return np.sqrt(np.sum((x - y) ** 2))

def dtw_distance(x,y):
  return dtw(x,y)


def cross_correlation_distance(x, y):
    x_norm = (x - np.mean(x)) / (np.std(x) * len(x))
    y_norm = (y - np.mean(y)) / np.std(y)

    cc = np.correlate(x_norm, y_norm, mode="full")
    max_correlation = np.max(np.abs(cc))
    distance = max(0.0, 1.0 - max_correlation)

    return distance
