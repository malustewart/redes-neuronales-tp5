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

def linear_regression(x: np.ndarray, y: np.ndarray):
    x_mean = x.mean()
    y_mean = y.mean()

    x_centered = x - x_mean
    y_centered = y - y_mean

    w1 = np.sum(x_centered*y_centered) / np.sum(x_centered**2)
    w0 = y_mean - w1 * x_mean

    return w0, w1

if __name__ == '__main__':
    height, weight = read_data('AlturaPeso.dat')
    N = len(height)
    n = 25000
    reps = 1
    for _ in range(reps):
        idx = random.sample(range(N), n)
        h = height[idx]
        w = weight[idx]
        w0, w1 = linear_regression(h, w)
        plot_data(h, w, w0, w1)
    plt.show()