import numpy as np
import pandas as pd
import math
import random
import time  # Track execution time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist

# Start the execution timer
start_time = time.time()

# CONFIGURATION & HYPERPARAMETERS
csv_path = "/Users/souviksarkarjr./Downloads/Experimental Datasets/Drug.csv"   # CSV File Path
n_clusters = 3
pop_size = 30          # P in the text
max_iter = 100         # T in the text
beta = 2               # Non-linear decay exponent
alpha = 0.01           # Levy flight step scaling
r_max = 1.0            # Adaptive control bounds
r_min = 0.0
epsilon = 1e-4         # Baseline probability exploration factor
fcm_m = 2.0            # Fuzzy fuzzifier parameter
fcm_iter = 100         # FCM max iterations

random.seed(42)
np.random.seed(42)

# DATA LOADING & PREPROCESSING
df = pd.read_csv(csv_path)
df = df.select_dtypes(include=[np.number])
if df.shape[1] == 0:
    raise ValueError("Dataset has no numeric columns!")

data = df.values
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

N, dim = data.shape
D = n_clusters * dim

# Replicate bounds across D dimensions
lb_base = np.min(data, axis=0)
ub_base = np.max(data, axis=0)
LB = np.tile(lb_base, n_clusters)
UB = np.tile(ub_base, n_clusters)

# --- HIGH-SPEED HELPER FUNCTIONS ---

def vector_to_centroids(Z):
    return Z.reshape(n_clusters, dim)

def centroids_to_vector(centroids):
    return centroids.flatten()

def assign_clusters(data, Z):
    centroids = Z.reshape(n_clusters, dim)
    distances = cdist(data, centroids, metric='sqeuclidean')
    return np.argmin(distances, axis=1)

def proxy_fitness(Z):
    centroids = Z.reshape(n_clusters, dim)
    distances = cdist(data, centroids, metric='euclidean')
    return -np.sum(np.min(distances, axis=1))

def fast_dunn_index(data, labels):
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < n_clusters:
        return 0
    
    clusters = [data[labels == k] for k in unique_clusters]
    
    max_diameter = 0
    for cluster in clusters:
        if len(cluster) > 1:
            max_dist = np.max(cdist(cluster, cluster, metric='euclidean'))
            if max_dist > max_diameter:
                max_diameter = max_dist
                
    if max_diameter == 0:
        return 0

    min_inter_dist = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            min_dist = np.min(cdist(clusters[i], clusters[j], metric='euclidean'))
            if min_dist < min_inter_dist:
                min_inter_dist = min_dist

    return min_inter_dist / max_diameter

def fast_xie_beni_index(data, labels, Z):
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

def compute_raw_indices(Z):
    labels = assign_clusters(data, Z)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < n_clusters:
        return -1.0, 100.0, 0.0, 0.0, 100.0
        
    s = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)
    ch = calinski_harabasz_score(data, labels)
    d = fast_dunn_index(data, labels)
    xb = fast_xie_beni_index(data, labels, Z)
    return s, db, ch, d, xb

def composite_fitness(Z, norm_bounds=None):
    s, db, ch, d, xb = compute_raw_indices(Z)
    
    f_s = s
    f_db = 1.0 / (1.0 + db)
    f_ch = ch / (1.0 + ch)
    f_d = d
    f_xb = 1.0 / (1.0 + xb)
    
    if norm_bounds:
        f_s = (f_s - norm_bounds['s_min']) / (norm_bounds['s_max'] - norm_bounds['s_min'] + 1e-6)
        f_db = (f_db - norm_bounds['db_min']) / (norm_bounds['db_max'] - norm_bounds['db_min'] + 1e-6)
        f_ch = (f_ch - norm_bounds['ch_min']) / (norm_bounds['ch_max'] - norm_bounds['ch_min'] + 1e-6)
        f_d = (f_d - norm_bounds['d_min']) / (norm_bounds['d_max'] - norm_bounds['d_min'] + 1e-6)
        f_xb = (f_xb - norm_bounds['xb_min']) / (norm_bounds['xb_max'] - norm_bounds['xb_min'] + 1e-6)

    return f_s + f_db + f_ch + f_d + f_xb

def levy_flight(dim_size, lam=1.5):
    sigma = (math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (math.gamma((1 + lam) / 2) * lam *
              2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = np.random.normal(0, sigma, dim_size)
    v = np.random.normal(0, 1, dim_size)
    return u / (np.abs(v) ** (1 / lam))

# --- PHASE 1: INITIALIZATION ---
raw_pool = []
r_chaotic = random.random()

for _ in range(pop_size):
    Z_primary = np.zeros(D)
    for m in range(D):
        r_chaotic = 4.0 * r_chaotic * (1.0 - r_chaotic)
        Z_primary[m] = LB[m] + r_chaotic * (UB[m] - LB[m])
        
    raw_pool.append(Z_primary)
    raw_pool.append(LB + UB - Z_primary)

raw_pool.sort(key=proxy_fitness, reverse=True)
population = np.array(raw_pool[:pop_size])

norm_bounds = {'s_min': -1.0, 's_max': 1.0, 'db_min': 0.0, 'db_max': 1.0,
               'ch_min': 0.0, 'ch_max': 1000.0, 'd_min': 0.0, 'd_max': 5.0, 'xb_min': 0.0, 'xb_max': 10.0}

fitness = np.array([composite_fitness(p, norm_bounds) for p in population])

best_idx = np.argmax(fitness)
best_Z = population[best_idx].copy()
best_score = fitness[best_idx]

strategy_success = np.ones(3)

# --- MAIN EVOLUTIONARY LOOP ---
for t in range(max_iter):
    r_t = r_max - ((t / max_iter) ** beta) * (r_max - r_min)
    probs = strategy_success / np.sum(strategy_success)
    
    print(f"Iteration {t+1}/{max_iter} | Best Composite Fitness: {best_score:.6f}")
    
    elite_count = max(1, int(0.1 * pop_size))
    elite_idx = np.argsort(fitness)[-elite_count:]
    elite_mean = np.mean(population[elite_idx], axis=0)
    
    strategies = np.random.choice(3, size=pop_size, p=probs)
    
    for i in range(pop_size):
        strategy = strategies[i]
        Xi = population[i]
        
        if strategy == 0:
            new_solution = Xi + r_t * (elite_mean - Xi)
        elif strategy == 1:
            a, b = np.random.choice([idx for idx in range(pop_size) if idx != i], 2, replace=False)
            new_solution = Xi + r_t * (population[a] - population[b])
        else:
            new_solution = Xi + alpha * levy_flight(D)
            
        new_solution = np.clip(new_solution, LB, UB)
        new_fit = composite_fitness(new_solution, norm_bounds)
        
        if new_fit > fitness[i]:
            population[i] = new_solution
            fitness[i] = new_fit
            strategy_success[strategy] += 1.0
            
            if new_fit > best_score:
                best_score = new_fit
                best_Z = new_solution.copy()
                
    strategy_success = np.maximum(strategy_success * 0.9, epsilon)

# --- FUZZY C-MEANS REFINEMENT ---
print("\nRefining solution using FCM...")

def fuzzy_c_means(data, initial_Z, m=2.0, max_iter=100, error=1e-5):
    centroids = initial_Z.reshape(n_clusters, dim).copy()
    
    for _ in range(max_iter):
        distances = cdist(data, centroids, metric='euclidean')
        distances = np.fmax(distances, 1e-10)

        power = 2.0 / (m - 1)
        temp = distances ** power
        denominator = np.sum((1.0 / temp), axis=1, keepdims=True)
        U = (1.0 / temp) / denominator

        U_m = U ** m
        new_centroids = (U_m.T @ data) / np.sum(U_m.T, axis=1, keepdims=True)

        if np.linalg.norm(new_centroids - centroids) < error:
            break
        centroids = new_centroids

    return centroids.flatten()

final_Z = fuzzy_c_means(data, best_Z, fcm_m, fcm_iter)
final_Z = np.clip(final_Z, LB, UB)
final_score = composite_fitness(final_Z, norm_bounds)

# Final Metric Reporting
s_score, db_score, ch_score, dunn, xb = compute_raw_indices(final_Z)
execution_time = time.time() - start_time

print("\n========== FINAL RESULTS (SAIFGO + FCM) ==========")
print("Optimized Composite Score:", final_score)
print("Silhouette Index (S):", s_score)
print("Calinski-Harabasz (CH):", ch_score)
print("Dunn Index (D):", dunn)
print("Davies-Bouldin (DB):", db_score)
print("Xie-Beni (XB):", xb)
print(f"Execution Time: {execution_time:.4f} seconds")