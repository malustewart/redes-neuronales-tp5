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

def plot_w_se_ci(W0, W1, SE0, SE1, w0_golden, w1_golden):
    fig, axs = plt.subplots(1,2)


    for i, (w0, se0) in enumerate(zip(W0, SE0)):
        l0 = w0 - 2*se0
        u0 = w0 + 2*se0
        color = 'k' if is_inside_confidence_interval(w0_golden, w0, se0) else 'r'
        axs[0].plot([i,i], [l0,u0], marker = "o", color=color)
    
    axs[0].axhline(w0_golden)
    
    # todo: poner bien las labels

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

if __name__ == '__main__':
    np.random.seed(12345)
    height, weight = read_data('AlturaPeso.dat')
    N = len(height)
    ns = [25, 100, 1000] #range(25,25000, 25)
    reps = 1000

    shape = (len(ns), reps)
    W0 = np.zeros(shape)
    W1 = np.zeros(shape)
    RSS = np.zeros(shape)
    TSS = np.zeros(shape)
    σ_sqr = np.zeros(shape)
    SE_sqr_w0 = np.zeros(shape)
    SE_sqr_w1 = np.zeros(shape)

    # Calcular valores "verdaderos" (la mejor estimacion posible)
    w0_golden, w1_golden, *_ = calc_predictors(height, weight)
    print(f"w0 golden = {w0_golden}\nw1 golden = {w1_golden}")

    # Estimar para diferentes n
    for i, n in enumerate(ns):
        reps = reps if n < N else 1
        for j in range(reps):
            idx = random.sample(range(N), n)
            h = height[idx]
            w = weight[idx]
            W0[i][j], W1[i][j], RSS[i][j], TSS[i][j], σ_sqr[i][j], SE_sqr_w0[i][j], SE_sqr_w1[i][j] = calc_predictors(h, w)

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

    plot_w_se_ci(W0[0][::50], W1[0][::50], SE_W0[0][::50], SE_W1[0][::50], w0_golden, w1_golden)
    plot_w_se_ci(W0[2][::50], W1[2][::50], SE_W0[2][::50], SE_W1[2][::50], w0_golden, w1_golden)



    plt.show()