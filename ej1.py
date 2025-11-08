import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Optional
import random
from tqdm import tqdm
import argparse

def read_data(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        height = np.array([float(row[1]) for row in reader])
        f.seek(0)
        weight = np.array([float(row[2]) for row in reader])
    return height, weight

def plot_data(height: np.ndarray, weight: np.ndarray, w0: float, w1: float, filename: Optional[str] = None):
    fig = plt.figure()
    plt.scatter(height, weight, c='blue', alpha=0.7, marker='x')
    plt.xlabel('Altura')
    plt.ylabel('Peso')
    plt.title('Peso vs Altura')
    plt.grid(True, linestyle='--', alpha=0.5)

    h_range = [height.min(), height.max()]
    w_range_est = [w0 + w1 * h for h in h_range]

    plt.plot(h_range, w_range_est)

    if filename:
        fig.savefig(filename)
        plt.close(fig)
    else:
        fig.show()

def params_to_str(params:dict, sep:str = " - "):
    return sep.join(f"{k}: {v}" for k,v in params.items())

def plot_w_histogram_2d(W0, W1, params, filepath=None):
    fig, ax = plt.subplots()
    _, _, _, im = ax.hist2d(W0, W1, bins=[50,50], density=True)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Frecuencia')
    ax.set_title(params_to_str(params))
    ax.set_xlabel("W0")
    ax.set_ylabel("W1")
    if filepath:
        fig.savefig(filepath)

def plot_w_histogram_1d(W0, W1, params):
    fig, ax = plt.subplots(2,1)
    ax[0].hist(W0, density=True)
    ax[0].set_xlabel("W0")
    ax[1].hist(W1, density=True)
    ax[1].set_xlabel("W1")
    ax[0].set_title(params_to_str(params))

def plot_w_se_ci(W0, W1, SE0, SE1, w0_golden, w1_golden, x_indices, x_label, params={}, filename=None):
    fig, axs = plt.subplots(2,1, figsize=(15,7))

    for i, (W,SE,w_golden) in enumerate(zip([W0, W1], [SE0, SE1], [w0_golden, w1_golden])):
        for w, se, x in zip(W, SE, x_indices):
            l0 = w - 2*se
            u0 = w + 2*se
            color = 'k' if is_inside_confidence_interval(w_golden, w, se) else 'r'
            axs[i].plot([x,x], [l0,u0], marker = "o", color=color, linewidth=0.6, markersize=1)
        axs[i].axhline(w_golden, color='k')
        axs[i].grid(linewidth=0.3, alpha=0.5)
        axs[i].set_ylabel(f"w{i}")
        axs[i].set_xlabel(x_label)

    fig.suptitle(params_to_str(params))

    fig.tight_layout()

    if filename:
        fig.savefig(filename)
        plt.close(fig)
    else:
        fig.show()

def linear_regression(x: np.ndarray, y: np.ndarray):
    x_mean = x.mean()
    y_mean = y.mean()

    x_centered = x - x_mean
    y_centered = y - y_mean

    w1 = np.sum(x_centered*y_centered) / np.sum(x_centered**2)
    w0 = y_mean - w1 * x_mean

    return w0, w1

def calc_predictors(h:np.ndarray, w:np.ndarray):
    N = len(h)
    w0, w1 = linear_regression(h, w)

    w_est = w0 + w1 * h

    RSS = np.sum((w-w_est)**2)
    TSS = np.sum((w-w.mean())**2)
    σ_sqr = RSS/(N-2)

    SE_sqr_w0 = σ_sqr * (1/N + h.mean()**2/np.sum((h-h.mean())**2))
    SE_sqr_w1 = σ_sqr * (1/np.sum((h-h.mean())**2))

    return w0, w1, RSS, TSS, σ_sqr, SE_sqr_w0, SE_sqr_w1

def is_inside_confidence_interval(value, mean, SE):
    return value > mean - 2*SE and value < mean + 2*SE

def generate_data(N, ns, reps):
    shape = (len(ns), reps)
    W0 = np.zeros(shape)
    W1 = np.zeros(shape)
    RSS = np.zeros(shape)
    TSS = np.zeros(shape)
    σ_sqr = np.zeros(shape)
    SE_sqr_w0 = np.zeros(shape)
    SE_sqr_w1 = np.zeros(shape)

    for i, n in enumerate(tqdm(ns)):
        reps = reps if n < N else 1
        for j in range(reps):
            idx = random.sample(range(N), n)
            h = height[idx]
            w = weight[idx]
            W0[i][j], W1[i][j], RSS[i][j], TSS[i][j], σ_sqr[i][j], SE_sqr_w0[i][j], SE_sqr_w1[i][j] = calc_predictors(h, w)
    return W0, W1, RSS, TSS, σ_sqr, SE_sqr_w0, SE_sqr_w1

def store_data(filename, W0, W1, RSS, TSS, σ_sqr, SE_sqr_w0, SE_sqr_w1):
    results = dict(
        W0=W0,
        W1=W1,
        RSS=RSS,
        TSS=TSS,
        σ_sqr=σ_sqr,
        SE_sqr_w0=SE_sqr_w0,
        SE_sqr_w1=SE_sqr_w1
    )
    np.savez_compressed(filename, *results)

def load_data(filename):
    data = np.load(filename)
    return data["W0"], data["W1"], data["RSS"], data["TSS"], data["σ_sqr"], data["SE_sqr_w0"], data["SE_sqr_w1"]

if __name__ == '__main__':
    np.random.seed(12345)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--generate", type=str, default=None,
                        help="Path to a new .npz file to be created.")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to an existing .npz file.")
    parser.add_argument("--analyze",
                        help="Analyze data.")
    args = parser.parse_args()

    height, weight = read_data('AlturaPeso.dat')
    N = len(height)
    ns = [10] + list(range(25,25000-25, 25))
    reps = 1000

    if args.generate:
        W0, W1, RSS, TSS, σ_sqr, SE_sqr_w0, SE_sqr_w1 = generate_data(N, ns, reps)
        store_data(args.generate, W0, W1, RSS, TSS, σ_sqr, SE_sqr_w0, SE_sqr_w1)

    if args.load:
        W0, W1, RSS, TSS, σ_sqr, SE_sqr_w0, SE_sqr_w1 = load_data(args.load)

    if args.analyze:
        # Calcular valores "verdaderos" (la mejor estimacion posible)
        w0_golden, w1_golden, *_ = calc_predictors(height, weight)
        print(f"w0 golden = {w0_golden}\nw1 golden = {w1_golden}")

        w0_mean = W0.mean(axis=1)
        w0_std = W0.std(axis=1)
        w1_mean = W1.mean(axis=1)
        w1_std = W1.std(axis=1)

        SE_W0 = np.sqrt(SE_sqr_w0)
        SE_W1 = np.sqrt(SE_sqr_w1)
        SE0_mean = SE_W0.mean(axis = 1)
        SE1_mean = SE_W1.mean(axis = 1)
        SE0_std = SE_W0.std(axis = 1)
        SE1_std = SE_W1.std(axis = 1)

        N_w0_intervals_contain_golden = np.array([sum(1 for w, se in zip(w0, se_w0) if is_inside_confidence_interval(w0_golden, w, se)) for w0, se_w0 in zip(W0, SE_W0)])
        N_w1_intervals_contain_golden = np.array([sum(1 for w, se in zip(w1, se_w1) if is_inside_confidence_interval(w1_golden, w, se)) for w1, se_w1 in zip(W1, SE_W1)])

        step = 1
        x_indices = range(0, len(W0[0]), step)

        for i, n in enumerate(ns[::200]):
            params  = {"n samples":n, "N reps": reps, "w0 CI misses": reps - N_w0_intervals_contain_golden[i], "w1 CI misses": reps - N_w1_intervals_contain_golden[i]}
            filename = f"figures/w_CI_n_{n}_reps_{reps}"
            plot_w_se_ci(W0[i][::step], W1[i][::step], SE_W0[i][::step], SE_W1[i][::step], w0_golden, w1_golden, x_indices, "Repetición", params=params, filename=filename)

        plt.show()