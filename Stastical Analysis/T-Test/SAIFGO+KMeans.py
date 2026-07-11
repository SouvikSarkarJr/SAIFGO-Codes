# ======================================================
#   SAIFGO + KMeans + INDEPENDENT T-TEST + LINE GRAPH
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind

# ==========================================================
# 1️⃣ HIGH-SPEED HELPER FUNCTIONS
# ==========================================================

def assign_clusters(data, Z, n_clusters, dim):
    centroids = Z.reshape(n_clusters, dim)
    distances = cdist(data, centroids, metric='sqeuclidean')
    return np.argmin(distances, axis=1)

def fast_dunn_index(data, labels, n_clusters):
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < n_clusters: return 0
    
    clusters = [data[labels == k] for k in unique_clusters]
    max_diameter = 0
    for cluster in clusters:
        if len(cluster) > 1:
            max_dist = np.max(cdist(cluster, cluster, metric='euclidean'))
            if max_dist > max_diameter: max_diameter = max_dist
                
    if max_diameter == 0: return 0

    min_inter_dist = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            min_dist = np.min(cdist(clusters[i], clusters[j], metric='euclidean'))
            if min_dist < min_inter_dist: min_inter_dist = min_dist
    return min_inter_dist / max_diameter

def fast_xie_beni_index(data, labels, Z, n_clusters, dim):
    N = data.shape[0]
    centroids = Z.reshape(n_clusters, dim)
    numerator = 0
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            numerator += np.sum((cluster_points - centroids[i])**2)

    centroid_dist = cdist(centroids, centroids, metric='sqeuclidean')
    np.fill_diagonal(centroid_dist, np.inf)
    min_centroid_dist = np.min(centroid_dist)
    return numerator / (N * min_centroid_dist) if min_centroid_dist != 0 else 0

def compute_raw_indices(data, Z, n_clusters, dim):
    labels = assign_clusters(data, Z, n_clusters, dim)
    unique_labels = np.unique(labels)
    if len(unique_labels) < n_clusters: return -1.0, 100.0, 0.0, 0.0, 100.0
        
    s = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)
    ch = calinski_harabasz_score(data, labels)
    d = fast_dunn_index(data, labels, n_clusters)
    xb = fast_xie_beni_index(data, labels, Z, n_clusters, dim)
    return s, db, ch, d, xb

def composite_fitness(data, Z, n_clusters, dim, norm_bounds=None):
    s, db, ch, d, xb = compute_raw_indices(data, Z, n_clusters, dim)
    f_s, f_db, f_ch, f_d, f_xb = s, 1.0 / (1.0 + db), ch / (1.0 + ch), d, 1.0 / (1.0 + xb)
    
    if norm_bounds:
        f_s = (f_s - norm_bounds['s_min']) / (norm_bounds['s_max'] - norm_bounds['s_min'] + 1e-6)
        f_db = (f_db - norm_bounds['db_min']) / (norm_bounds['db_max'] - norm_bounds['db_min'] + 1e-6)
        f_ch = (f_ch - norm_bounds['ch_min']) / (norm_bounds['ch_max'] - norm_bounds['ch_min'] + 1e-6)
        f_d = (f_d - norm_bounds['d_min']) / (norm_bounds['d_max'] - norm_bounds['d_min'] + 1e-6)
        f_xb = (f_xb - norm_bounds['xb_min']) / (norm_bounds['xb_max'] - norm_bounds['xb_min'] + 1e-6)

    return f_s + f_db + f_ch + f_d + f_xb

def proxy_fitness(data, Z, n_clusters, dim):
    centroids = Z.reshape(n_clusters, dim)
    distances = cdist(data, centroids, metric='euclidean')
    return -np.sum(np.min(distances, axis=1))

def levy_flight(dim_size, lam=1.5):
    sigma = (math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (math.gamma((1 + lam) / 2) * lam *
              2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = np.random.normal(0, sigma, dim_size)
    v = np.random.normal(0, 1, dim_size)
    return u / (np.abs(v) ** (1 / lam))

# ==========================================================
# 2️⃣ SAIFGO + K-MEANS CLASS
# ==========================================================

class SAIFGO_KMeans_Clustering:
    def __init__(self, n_clusters=3, pop_size=20, max_iter=30, beta=2, alpha=0.01, 
                 r_max=1.0, r_min=0.0, epsilon=1e-4):
        self.n_clusters = n_clusters
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.beta = beta
        self.alpha = alpha
        self.r_max = r_max
        self.r_min = r_min
        self.epsilon = epsilon
        self.norm_bounds = {'s_min': -1.0, 's_max': 1.0, 'db_min': 0.0, 'db_max': 1.0,
                            'ch_min': 0.0, 'ch_max': 1000.0, 'd_min': 0.0, 'd_max': 5.0, 'xb_min': 0.0, 'xb_max': 10.0}

    def fit(self, X):
        N, dim = X.shape
        D = self.n_clusters * dim
        lb_base, ub_base = np.min(X, axis=0), np.max(X, axis=0)
        LB, UB = np.tile(lb_base, self.n_clusters), np.tile(ub_base, self.n_clusters)

        raw_pool = []
        r_chaotic = random.random()

        for _ in range(self.pop_size):
            Z_primary = np.zeros(D)
            for m in range(D):
                r_chaotic = 4.0 * r_chaotic * (1.0 - r_chaotic)
                Z_primary[m] = LB[m] + r_chaotic * (UB[m] - LB[m])
            raw_pool.append(Z_primary)
            raw_pool.append(LB + UB - Z_primary)

        raw_pool.sort(key=lambda z: proxy_fitness(X, z, self.n_clusters, dim), reverse=True)
        population = np.array(raw_pool[:self.pop_size])

        fitness = np.array([composite_fitness(X, p, self.n_clusters, dim, self.norm_bounds) for p in population])
        best_idx = np.argmax(fitness)
        best_Z = population[best_idx].copy()
        best_score = fitness[best_idx]
        strategy_success = np.ones(3)

        for t in range(self.max_iter):
            r_t = self.r_max - ((t / self.max_iter) ** self.beta) * (self.r_max - self.r_min)
            probs = strategy_success / np.sum(strategy_success)
            
            elite_count = max(1, int(0.1 * self.pop_size))
            elite_idx = np.argsort(fitness)[-elite_count:]
            elite_mean = np.mean(population[elite_idx], axis=0)
            strategies = np.random.choice(3, size=self.pop_size, p=probs)
            
            for i in range(self.pop_size):
                strategy = strategies[i]
                Xi = population[i]
                
                if strategy == 0:
                    new_solution = Xi + r_t * (elite_mean - Xi)
                elif strategy == 1:
                    a, b = np.random.choice([idx for idx in range(self.pop_size) if idx != i], 2, replace=False)
                    new_solution = Xi + r_t * (population[a] - population[b])
                else:
                    new_solution = Xi + self.alpha * levy_flight(D)
                    
                new_solution = np.clip(new_solution, LB, UB)
                new_fit = composite_fitness(X, new_solution, self.n_clusters, dim, self.norm_bounds)
                
                if new_fit > fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fit
                    strategy_success[strategy] += 1.0
                    if new_fit > best_score:
                        best_score = new_fit
                        best_Z = new_solution.copy()
                        
            strategy_success = np.maximum(strategy_success * 0.9, self.epsilon)

        print("SAIFGO Optimization Completed. Refining with K-Means...")
        initial_centroids = best_Z.reshape(self.n_clusters, dim)
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=initial_centroids,
            n_init=1,
            max_iter=300,
            random_state=42
        )
        kmeans.fit(X)

        self.best_centers = kmeans.cluster_centers_
        self.labels_ = kmeans.labels_
        return self


# ======================================================
# 3️⃣ MAIN EXECUTION
# ======================================================

if __name__ == "__main__":

    file_path = "dataset.csv"   # <-- Change this to your dataset path
    n_clusters = 3

    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}. Please update the path.")
    else:
        df = pd.read_csv(file_path)

        X = df.select_dtypes(include=[np.number])
        X = X.drop(columns=["Id", "ID", "id"], errors="ignore")
        features = X.columns.tolist()

        # Scale data to conform to the algorithm's constraints
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model = SAIFGO_KMeans_Clustering(
            n_clusters=n_clusters,
            pop_size=20,
            max_iter=30
        )

        model.fit(X_scaled)
        labels = model.labels_

        df["Cluster"] = labels
        print("\n================ CLUSTERING DONE ================")
        print(df["Cluster"].value_counts())
        print("=================================================\n")

        # ======================================================
        # INDEPENDENT T-TEST (Welch)
        # ======================================================

        clusters = np.unique(labels)

        if len(clusters) < 2:
            raise ValueError("Need at least 2 clusters for T-Test.")

        cluster_0 = X_scaled[labels == clusters[0]]
        cluster_1 = X_scaled[labels == clusters[1]]

        p_values = []

        print("============= T-TEST RESULTS =============")

        for i in range(len(features)):
            stat, p_value = ttest_ind(
                cluster_0[:, i],
                cluster_1[:, i],
                equal_var=False
            )

            p_values.append(p_value)

            print(f"\nFeature: {features[i]}")
            print(f"T-Statistic = {stat:.4f}")
            print(f"P-value     = {p_value:.6f}")

            if p_value < 0.05:
                print("Significant Difference")
            else:
                print("Not Significant")

        print("==========================================\n")

        # ======================================================
        # GENERATE LINE GRAPH
        # ======================================================

        plt.figure()
        plt.plot(features, p_values, marker='o')
        plt.axhline(y=0.05, color='r', linestyle='--')
        plt.xticks(rotation=45)
        plt.title("T-Test (Cluster 0 vs Cluster 1) - SAIFGO+KMeans")
        plt.xlabel("Features")
        plt.ylabel("P-values")
        plt.tight_layout()

        output_file = "SAIFGO+KMeans_TTest.jpg"
        plt.savefig(output_file, format="jpg")
        plt.close()

        print(f"Graph saved successfully as: {output_file}")