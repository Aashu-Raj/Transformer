import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import math
import pandas as pd

# Enable GPU and set memory growth to prevent full memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow will use GPU(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, running on CPU.")

from MemoryTF2Transformer import MemoryDNN
from ResourceAllocationtf import Algo1_NUM_VariableEnergy


# === Set Random Seeds for Reproducibility ===
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === Create directory for CSV and plots ===
csv_dir = "frame_logs_transformer"
os.makedirs(csv_dir, exist_ok=True)

# === User Count Generation ===
def generate_user_count_sequence(n, min_users, max_users, max_changes):
    change_points = sorted(random.sample(range(1, n), min(max_changes, n - 1)))
    change_points = [0] + change_points + [n]
    user_counts = []
    for i in range(len(change_points) - 1):
        user_value = random.randint(min_users, max_users)
        user_counts.extend([user_value] * (change_points[i + 1] - change_points[i]))
    return np.array(user_counts)

def load_or_create_user_counts(n, min_users, max_users, max_changes, filename="user_count_list.npy"):
    if os.path.exists(filename):
        user_counts = np.load(filename)
    else:
        user_counts = generate_user_count_sequence(n, min_users, max_users, max_changes)
        np.save(filename, user_counts)
    return user_counts

# === Rician Fading Channel Model ===
def racian_mec(h, factor):
    n = len(h)
    beta = np.sqrt(h * factor)
    sigma = np.sqrt(h * (1 - factor) / 2)
    x = sigma * np.random.randn(n) + beta
    y = sigma * np.random.randn(n)
    return np.power(x, 2) + np.power(y, 2)

# === Save array as CSV with optional header ===
def save_array_as_csv(arr, filename, header=None):
    path = os.path.join(csv_dir, filename)
    # If arr is 1D, reshape to 2D for consistent saving
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    np.savetxt(path, arr, delimiter=",", header=",".join(header) if header else "", comments='')

# === Plot with rolling mean and save ===
def plot_array(data_array, rolling_intv=50, ylabel='Value', save_path=None):
    df = pd.DataFrame(data_array)
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.plot(
        np.arange(len(data_array)) + 1,
        np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values),
        linewidth=2,
        label='Rolling Mean'
    )
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # === Simulation Parameters ===
    n = 10000
    min_users = 5
    N = 30
    max_changes = 20
    lambda_param = 3
    nu = 1000
    Delta = 32
    V = 20
    Memory_capacity = 1024
    decoder_mode = 'OP'
    CHFACT = 10**10

    # Load or generate user count sequence
    user_count_list = load_or_create_user_counts(n, min_users, N, max_changes)

    # Initialize state arrays
    channel = np.zeros((n, N))
    dataA = np.zeros((n, N))
    Q = np.zeros((n, N))
    Y = np.zeros((n, N))
    Obj = np.zeros(n)
    energy_arr = np.zeros((n, N))
    rate_arr = np.zeros((n, N))

    # Thresholds and weights
    energy_thresh = np.ones(N) * 0.08
    w = np.array([1.5 if i % 2 == 0 else 1 for i in range(N)])
    arrival_lambda = lambda_param * np.ones(N)

    # Channel pathloss calculation
    dist_v = np.linspace(start=120, stop=255, num=N)
    Ad, fc = 3, 915e6
    loss_exponent = 3
    light = 3e8
    h0 = np.array([Ad * (light / (4 * math.pi * fc * dist_v[j])) ** loss_exponent for j in range(N)])

    # Initialize MemoryDNN with Transformer
    mem = MemoryDNN(net=[N * 3, N], max_users=N, learning_rate=0.001,
                    training_interval=20, batch_size=128, memory_size=Memory_capacity)

    mode_his, k_idx_his = [], []
    start_time = time.time()

    for i in range(n):
        current_N = user_count_list[i]

        if i % (n // 10) == 0:
            print(f"Time Frame {i}/{n} | Active Users: {current_N} | {100 * i / n:.1f}%")

        # Compute K (number of users to select) safely
        if i > 0 and i % Delta == 0:
            if Delta > 1 and len(k_idx_his) >= Delta:
                recent_k = np.array(k_idx_his[-Delta:]) % current_N
                max_k = recent_k.max() + 1
            else:
                max_k = 1
        else:
            max_k = k_idx_his[-1] + 1 if k_idx_his else 1

        K = min(max_k, current_N)

        # Generate current channel gains and data arrivals
        h_tmp = racian_mec(h0[:current_N], 0.3)
        h_curr = h_tmp * CHFACT
        channel[i, :current_N] = h_curr
        dataA[i, :current_N] = np.random.exponential(arrival_lambda[:current_N])

        # Update queues and energy state
        if i > 0:
            Q[i, :current_N] = Q[i - 1, :current_N] + dataA[i - 1, :current_N] - rate_arr[i - 1, :current_N]
            Q[i, :current_N] = np.maximum(Q[i, :current_N], 0)
            Y[i, :current_N] = np.maximum(
                Y[i - 1, :current_N] + (energy_arr[i - 1, :current_N] - energy_thresh[:current_N]) * nu, 0)

        # Prepare input vector for NN (flattened concatenation)
        nn_input_raw = np.hstack([
            h_curr,
            Q[i, :current_N] / 10000,
            Y[i, :current_N] / 10000
        ])

        # Pad input vector to fixed size net[0] if needed
        if len(nn_input_raw) < mem.net[0]:
            nn_input_raw = np.hstack([nn_input_raw, np.zeros(mem.net[0] - len(nn_input_raw))])
        elif len(nn_input_raw) > mem.net[0]:
            nn_input_raw = nn_input_raw[:mem.net[0]]

        # Decode modes with MemoryDNN Transformer
        m_pred, m_list = mem.decode(nn_input_raw, K, decoder_mode)

        r_list, v_list = [], []
        for m in m_list:
            # Slice mode vector to current active users
            m_slice = np.array(m[:current_N])
            w_slice = w[:current_N]
            Q_slice = Q[i, :current_N]
            Y_slice = Y[i, :current_N]

            res = Algo1_NUM_VariableEnergy(m_slice, h_curr, w_slice, Q_slice, Y_slice, current_N, V)
            r_list.append(res)
            v_list.append(res[0])

        best_idx = np.argmax(v_list)
        k_idx_his.append(best_idx)
        mem.encode(nn_input_raw, m_list[best_idx])
        mode_his.append(m_list[best_idx])

        Obj[i] = r_list[best_idx][0]
        rate_arr[i, :current_N] = r_list[best_idx][1]
        energy_arr[i, :current_N] = r_list[best_idx][2]

    total_time = time.time() - start_time
    mem.plot_cost()

    # Save arrays as CSV with headers
    user_headers = [f'user_{i+1}' for i in range(N)]

    save_array_as_csv(channel / CHFACT, "channel.csv", header=user_headers)
    save_array_as_csv(dataA, "data_arrival.csv", header=user_headers)
    save_array_as_csv(Q, "data_queue.csv", header=user_headers)
    save_array_as_csv(Y, "energy_queue.csv", header=user_headers)
    save_array_as_csv(rate_arr, "rate.csv", header=user_headers)
    save_array_as_csv(energy_arr, "energy_consumption.csv", header=user_headers)

    # Frame-wise arrays with single header
    save_array_as_csv(Obj, "objective.csv", header=["objective"])
    save_array_as_csv(user_count_list, "user_count_list.csv", header=["user_count"])

    # Computation rate per frame
    comp_rate = np.sum(rate_arr, axis=1)
    save_array_as_csv(comp_rate, "computational_rate.csv", header=["comp_rate"])

    # Save plots
    plot_array(np.sum(Q, axis=1) / user_count_list, rolling_intv=50,
               ylabel='Average Data Queue',
               save_path=os.path.join(csv_dir, "avg_queue_plot.png"))

    plot_array(np.sum(energy_arr, axis=1) / user_count_list, rolling_intv=50,
               ylabel='Average Energy Consumption',
               save_path=os.path.join(csv_dir, "avg_energy_plot.png"))

    plot_array(comp_rate, rolling_intv=50,
               ylabel='Computational Rate',
               save_path=os.path.join(csv_dir, "computational_rate_plot.png"))

    # Save as .mat file
    sio.savemat("result_transformer_dynamic_users.mat", {
        'input_h': channel / CHFACT,
        'data_arrival': dataA,
        'data_queue': Q,
        'energy_queue': Y,
        'off_mode': mode_his,
        'rate': rate_arr,
        'energy_consumption': energy_arr,
        'data_rate': rate_arr,
        'objective': Obj,
        'user_count_list': user_count_list
    })

    print(f"Results saved in result_transformer_dynamic_users.mat")
    print(f"Average time per frame: {total_time/n:.4f} seconds") 