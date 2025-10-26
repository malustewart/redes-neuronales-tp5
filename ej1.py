import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Optional

def read_data(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        height = np.array([row[1] for row in reader])
        f.seek(0)
        weight = np.array([row[2] for row in reader])
    return height, weight

def plot_data(height: np.ndarray, weight: np.ndarray, filename: Optional[str] = None):
    fig = plt.figure()
    plt.scatter(height, weight, c='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Altura')
    plt.ylabel('Peso')
    plt.title('Peso vs Altura')
    plt.grid(True, linestyle='--', alpha=0.5)

    if filename:
        fig.savefig(filename)
        plt.close(fig)
    else:
        fig.show()


if __name__ == '__main__':
    h, w = read_data('AlturaPeso.dat')
    plot_data(h, w)