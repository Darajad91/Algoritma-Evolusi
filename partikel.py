import numpy as np

# Fungsi aktivasi
def fitness_function(x):
    return (x - 5)**2 + 10

# Parameter PSO
w = 0.5
c1 = 1.0 
c2 = 1.0 

# Ukuran Populasi dan Jumlah Bit
POP_SIZE = 3
BITS = 4

# ditentukan nilai r1 r2 dan random check
r1 = np.array([0.2, 0.4, 0.6, 0.8])
r2 = np.array([0.9, 0.7, 0.5, 0.3])
random_check = np.array([0.7, 0.2, 0.8, 0.4])

# Inisialisasi Posisi dan Kecepatan Awal ( iterasi 0)
X = np.array([
    [0, 0, 1, 0],  
    [1, 0, 0, 0],  
    [1, 1, 0, 0]  
])
# kecepatan awal diinisialisasi nol
V = np.zeros(X.shape)

# Fungsi konversi biner ke desimal
def binary_to_decimal(binary_array):
    return np.dot(binary_array, 2**np.arange(binary_array.size)[::-1])

# Fungsi Sigmoid
def sigmoid(v):
    return 1 / (1 + np.exp(-v))

# --- Inisialisasi Pbest dan Gbest (t=0) ---

P_best_pos = X.copy()
X_decimal = np.array([binary_to_decimal(x) for x in X])
P_best_fitness = fitness_function(X_decimal)
current_fitness = P_best_fitness.copy()

# mencari Gbest terbesar
gbest_idx = np.argmax(P_best_fitness)
G_best_pos = P_best_pos[gbest_idx].copy()
G_best_fitness = P_best_fitness[gbest_idx]

print("--- INISIALISASI (t=0) ---")
print(f"Posisi Awal (X):\n{X}")
print(f"Fitness Awal: {current_fitness}")
print(f"Pbest Fitness: {P_best_fitness}")
print(f"Gbest Posisi: {G_best_pos}, Gbest Fitness: {G_best_fitness}\n")

# --- Looping Iterasi 2 ---
NUM_ITERATIONS = 2

for t in range(NUM_ITERATIONS):
    print(f"================ ITERASI t={t+1} ================")
    
    # simpan posisi baru
    X_new = np.zeros(X.shape, dtype=int)

    for i in range(POP_SIZE):
        
        # update kecepatan V
        diff_p = P_best_pos[i] - X[i]
        diff_g = G_best_pos - X[i]
        
        # Komponen update
        cognitive = c1 * r1 * diff_p
        social = c2 * r2 * diff_g
        
        # Update V (Vector-based)
        V[i] = w * V[i] + cognitive + social

        # --- Update Posisi (X) menggunakan Sigmoid ---
        
        # Hitung Probabilitas
        S_v = sigmoid(V[i])
        
        # Update X biner: X_new[i, j] = 1 jika rand < Sigmoid(V[i, j])
        X_new[i] = np.where(random_check < S_v, 1, 0)
        
    X = X_new.copy()
    X_decimal = np.array([binary_to_decimal(x) for x in X])
    current_fitness = fitness_function(X_decimal)
    
    # --- Update Pbest dan Gbest ---
    
    # Update Pbest (Mencari Maksimum: fitness baru > fitness lama)
    for i in range(POP_SIZE):
        if current_fitness[i] > P_best_fitness[i]:
            P_best_fitness[i] = current_fitness[i]
            P_best_pos[i] = X[i].copy()
            
    # Update Gbest (Mencari Maksimum: fitness terbesar di antara Pbest)
    gbest_idx = np.argmax(P_best_fitness)
    G_best_pos = P_best_pos[gbest_idx].copy()
    G_best_fitness = P_best_fitness[gbest_idx]

    print(f"Posisi Baru (X): {X_decimal} -> \n{X}")
    print(f"Kecepatan Baru (V):\n{V.round(4)}")
    print(f"Pbest Fitness Baru: {P_best_fitness}")
    print(f"Gbest Posisi Akhir Iterasi {t+1}: {G_best_pos}, Gbest Fitness: {G_best_fitness}")
    print("-------------------------------------------\n")

print("--- HASIL AKHIR ---")
print(f"Gbest Posisi (Biner): {G_best_pos}")
print(f"Gbest Posisi (Desimal): {binary_to_decimal(G_best_pos)}")
print(f"Gbest Fitness (Maksimum): {G_best_fitness}")