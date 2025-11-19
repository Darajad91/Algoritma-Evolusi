import math

# Helper
def bits_to_x(b):
    return b[0]*8 + b[1]*4 + b[2]*2 + b[3]*1

def fitness(x):
    return (x-5)**2 + 10

# Parameter
omega = 0.5
c1 = 1.0
c2 = 1.0

r1 = [0.2, 0.4, 0.6, 0.8]
r2 = [0.9, 0.7, 0.5, 0.3]
random_check = [0.7, 0.2, 0.8, 0.4]

# Inisialisasi
positions = [
    [0,0,1,0],  # x=2
    [1,0,0,0],  # x=8
    [1,1,0,0],  # x=12
]
velocities = [[0.0]*4 for _ in range(3)]
pbest_positions = [p.copy() for p in positions]
pbest_values = [fitness(bits_to_x(p)) for p in pbest_positions]
gbest_idx = min(range(3), key=lambda i: pbest_values[i])
gbest_position = pbest_positions[gbest_idx].copy()
gbest_value = pbest_values[gbest_idx]

print("Inisialisasi:")
for i,p in enumerate(positions):
    print(f" Partikel {i+1}: bits={p} x={bits_to_x(p)} fitness={pbest_values[i]}")
print(f" Gbest awal: idx={gbest_idx+1} bits={gbest_position} x={bits_to_x(gbest_position)} fitness={gbest_value}")
print("-"*50)

# Lakukan 2 iterasi penuh
for it in range(1,3):
    print(f"Iterasi {it}:")
    # update velocity & posisi
    new_vels = []
    new_pos = []
    for i in range(3):
        v_new = [0.0]*4
        for j in range(4):
            cog = c1 * r1[j] * (pbest_positions[i][j] - positions[i][j])  # =0 pada t=0
            soc = c2 * r2[j] * (gbest_position[j] - positions[i][j])
            v_new[j] = omega * velocities[i][j] + cog + soc
        # sigmoid
        s = [1.0 / (1.0 + math.exp(-vv)) for vv in v_new]
        # update bit biner
        pos_new = [1 if s[j] > random_check[j] else 0 for j in range(4)]
        new_vels.append(v_new)
        new_pos.append(pos_new)
        print(f" Partikel {i+1}: v_old={velocities[i]} -> v_new={[round(x,4) for x in v_new]}, sigmoid={[round(x,4) for x in s]}")
        print(f"            pos_old={positions[i]} -> pos_new={pos_new} (x={bits_to_x(pos_new)})")
    # assign
    velocities = new_vels
    positions = new_pos

    # update pbest dan gbest
    for i in range(3):
        val = fitness(bits_to_x(positions[i]))
        if val < pbest_values[i]:
            pbest_values[i] = val
            pbest_positions[i] = positions[i].copy()
    gbest_idx = min(range(3), key=lambda i: pbest_values[i])
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_value = pbest_values[gbest_idx]

    print(f" Setelah iterasi {it}: Gbest idx={gbest_idx+1}, bits={gbest_position}, x={bits_to_x(gbest_position)}, fitness={gbest_value}")
    print("-"*50)
