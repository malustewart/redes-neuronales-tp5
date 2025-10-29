import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Optional
import random

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

if __name__ == '__main__':
    height, weight = read_data('AlturaPeso.dat')
    N = len(height)
    ns = [25, 100, 1000] #range(25,25000, 25)
    reps = 1000
    W0_means = np.zeros(len(ns))
    W0_stds = np.zeros(len(ns))
    W1_means = np.zeros(len(ns))
    W1_stds = np.zeros(len(ns))

    for i, n in enumerate(ns):
        reps = reps if n < N else 1
        W0 = np.zeros(reps)
        W1 = np.zeros(reps)
        RSS = np.zeros(reps)
        TSS = np.zeros(reps)
        σ_sqr = np.zeros(reps)
        SE_sqr_w0 = np.zeros(reps)
        SE_sqr_w1 = np.zeros(reps)
        for j in range(reps):
            idx = random.sample(range(N), n)
            h = height[idx]
            w = weight[idx]
            w0, w1, rss, tss, sigma_sqr, se_sqr_w0, se_sqr_w1 = calc_predictors(h, w)
            W0[j] = w0
            W1[j] = w1
            RSS[j] = rss
            TSS[j] = tss
            σ_sqr[j] = sigma_sqr
            SE_sqr_w0[j] = se_sqr_w0
            SE_sqr_w1[j] = se_sqr_w1

        w0_mean = W0.mean()
        w1_mean = W1.mean()
        w0_std = W0.std()
        w1_std = W1.std()
        print(f"n={n}: w0 = {w0_mean}+-{w0_std}")
        print(f"n={n}: w1 = {w1_mean}+-{w1_std}")
        W0_means[i] = w0_mean
        W0_stds[i] = w0_std
        W1_means[i] = w1_mean
        W1_stds[i] = w1_std



        plot_w_histogram_2d(W0, W1, {"n":n, "N_rep": reps})
        plot_w_histogram_1d(W0, W1, {"n":n, "N_rep": reps})




    plt.show()