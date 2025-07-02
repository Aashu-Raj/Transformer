import numpy as np
from scipy.special import lambertw
from scipy.optimize import linprog


def Algo1_NUM_VariableEnergy(mode, h, w, Q, Y, num_users, V=20, dynamic_factor=0.1):
    ch_fact = 10 ** 10
    d_fact = 10 ** 6
    Y_factor = 10
    Y = Y * Y_factor
    phi = 100
    W = 2
    k_factor = (10 ** (-26)) * (d_fact ** 3)
    vu = 1.1

    N0 = W * d_fact * (10 ** (-17.4)) * (10 ** (-3)) * ch_fact
    P_max = 0.1
    f_max = 300

    N = num_users
    if len(w) == 0:
        w = np.ones((N))

    a = np.ones((N))
    q = Q
    for i in range(N):
        a[i] = Q[i] + V * w[i]

    energy = np.zeros((N))
    rate = np.zeros((N))
    f0_val = 0

    # Local computing
    idx0 = np.where(mode == 0)[0]
    M0 = len(idx0)

    if M0 != 0:
        for i in range(M0):
            tmp_id = idx0[i]
            y_val = Y[tmp_id]
            a_val = a[tmp_id]
            q_val = q[tmp_id]

            if y_val == 0:
                f_opt = np.minimum(phi * q_val, f_max)
            else:
                tmp1 = np.sqrt(a_val / (3 * phi * k_factor * y_val))
                tmp2 = np.minimum(phi * q_val, f_max)
                f_opt = np.minimum(tmp1, tmp2)

            # Inject variability in energy modeling
            adaptive_noise = 1 + dynamic_factor * np.random.randn()
            f_opt = np.clip(f_opt * adaptive_noise, 0, f_max)

            e = k_factor * (f_opt ** 3)
            r = f_opt / phi

            energy[tmp_id] = e
            rate[tmp_id] = r
            f0_val += a_val * r - y_val * e

    # Offloading
    idx1 = np.where(mode == 1)[0]
    M1 = len(idx1)

    if M1 == 0:
        f1_val = 0
    else:
        Y1 = Y[idx1]
        a1 = a[idx1]
        q1 = q[idx1]
        h1 = h[idx1]

        SNR = h1 / N0
        R_max = W / vu * np.log2(1 + SNR * P_max)

        rat = np.zeros((M1))
        e_ratio = np.zeros((M1))
        parac = np.zeros((M1))
        c = np.zeros((M1))
        tau1 = np.zeros((M1))

        lb = 0
        ub = 1e4
        delta0 = 1

        while np.abs(ub - lb) > delta0:
            mu = (lb + ub) / 2
            for i in range(M1):
                if Y1[i] == 0:
                    rat[i] = R_max[i]
                else:
                    A = 1 + mu / (Y1[i] * P_max)
                    A = np.minimum(A, 20)
                    tmpA = np.real(lambertw(-A * np.exp(-A)))
                    tmp1 = np.minimum(-A / tmpA, 1e20)
                    snr0 = 1 / P_max * (tmp1 - 1)
                    if SNR[i] <= snr0:
                        rat[i] = R_max[i]
                    else:
                        z1 = np.exp(-1) * (mu * SNR[i] / Y1[i] - 1)
                        rat[i] = (np.real(lambertw(z1)) + 1) * W / (np.log(2) * vu)

                e_ratio[i] = 1 / SNR[i] * (2 ** (rat[i] * vu / W) - 1)
                parac[i] = a1[i] - mu / rat[i] - (Y1[i] * e_ratio[i]) / rat[i]
                c[i] = q1[i] if parac[i] > 0 else 0
                tau1[i] = c[i] / rat[i]

            if np.sum(tau1) > 1:
                lb = mu
            else:
                ub = mu

        para_e = Y1 * e_ratio / rat
        para = a1 - para_e
        tau_fact = 1 / rat

        A_matrix = np.zeros((2 * M1 + 1, M1))
        b = np.zeros((2 * M1 + 1))
        A_matrix[:M1, :] = np.eye(M1)
        A_matrix[M1:2 * M1, :] = -np.eye(M1)
        A_matrix[2 * M1, :] = tau_fact
        b[:M1] = q1
        b[2 * M1] = 1

        res = linprog(-para, A_ub=A_matrix, b_ub=b)
        r1 = np.maximum(res.x, 0)

        f1_val = 0
        for i in range(M1):
            tmp_id = idx1[i]
            tau_i = r1[i] / rat[i]
            # Add variability in energy cost based on load + random factor
            variability = 1 + dynamic_factor * np.random.randn()
            energy[tmp_id] = e_ratio[i] * tau_i * variability
            rate[tmp_id] = r1[i]
            f1_val += a1[i] * rate[tmp_id] - Y1[i] * energy[tmp_id]

    f_val = f0_val + f1_val
    return np.round(f_val, 6), np.round(rate, 6), np.round(energy, 6)  