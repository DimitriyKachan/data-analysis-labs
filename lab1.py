import numpy as np
import matplotlib.pyplot as plt


def distance(x, y):
    return abs(np.linalg.norm(x-y))


def plot_k_means(X, K, max_iter=20):
    N, D = X.shape
    M = np.zeros((K, D))
    R = np.zeros((N, K))

    for k in range(K):
        M[k] = X[np.random.choice(N)]

    for i in range(max_iter):
        for k in range(K):
            for n in range(N):
                R[n, k] = np.exp(-distance(M[k], X[n])) / sum(np.exp(-distance(M[j], X[n])) for j in range(K))

    for k in range(K):
        M[k] = R[:, k].dot(X) / R[:, k].sum()

    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.scatter(M[:, 0], M[:, 1], c="black")
    plt.show()


def main():
    D = 2
    s = 4
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 900
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    K = 3
    plot_k_means(X, K)

    K = 5
    plot_k_means(X, K, max_iter=30)


if __name__ == "__main__":
    main()