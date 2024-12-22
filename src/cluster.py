"""
Used to cluster the set of time series of logprobs, probabilities or log-odds over layers

#probs is list of arrays/lists containing probs or logprobs across layers
labels = cluster(probs, num_clusters=4, plot=True)
"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def normalize(A):
    return [i / max(i) for i in A]


def smooth(A, window_size=5):
    weights = np.ones(window_size) / window_size
    # Simple moving average
    smoothed_A_SMA = np.convolve(A, weights, mode="valid")
    return smoothed_A_SMA


def kmeans(time_series_data, num_clusters=4, random_state=0):

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    labels = kmeans.fit_predict(time_series_data)

    return labels


def cluster(
    probs,
    num_clusters=4,
    plot=True,
    random_state=0,
    smooth_window_size=1,
    norm_by_max=False,
):

    probs = [smooth(i, smooth_window_size) for i in probs]

    if norm_by_max:
        probs = normalize(probs)

    labels = kmeans(np.array(probs), num_clusters, random_state)

    labelnames = list(set(labels.tolist()))
    if plot and num_clusters == 4:

        nlayers = len(probs[0])
        means = []
        stds = []
        for label in labelnames:
            mean = [
                np.mean([x[layer] for xx, x in enumerate(probs) if labels[xx] == label])
                for layer in range(nlayers)
            ]
            std = [
                np.std([x[layer] for xx, x in enumerate(probs) if labels[xx] == label])
                for layer in range(nlayers)
            ]
            means.append(mean)
            stds.append(np.array(std))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        axes = axes.flatten()
        colors = ["b", "g", "r", "m"]

        for i in range(4):
            axes[i].plot(range(nlayers), means[i], label=f"mean")
            axes[i].fill_between(
                range(nlayers),
                np.array(means[i]) - stds[i],
                np.array(means[i]) + stds[i],
                alpha=0.2,
                color=colors[i],
            )
            axes[i].set_title(f"Cluster {i}")
            axes[i].set_xlabel("Layers")
            axes[i].set_ylabel("Values")
            axes[i].legend()
            axes[i].grid()

        plt.tight_layout()
        plt.show()

    return labels
