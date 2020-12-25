
from time import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import euclidean_distances
density_param = {'density': True}

def check_rp(data, n_components_range = np.array([300])):
    n_samples, n_features = data.shape
    print("Embedding %d samples with dim %d using various random projections"
          % (n_samples, n_features))
    dists = euclidean_distances(data, squared=True).ravel()
    # select only non-identical samples pairs
    nonzero = dists != 0
    dists = dists[nonzero]
    for n_components in n_components_range:
        t0 = time()
        rp = GaussianRandomProjection(n_components=n_components,random_state = 0)
        projected_data = rp.fit_transform(data)
        print("Projected %d samples from %d to %d in %0.3fs"
              % (n_samples, n_features, n_components, time() - t0))
        if hasattr(rp, 'components_'):
            n_bytes = rp.components_.data.nbytes
    #         n_bytes += rp.components_.indices.nbytes
            print("Random matrix with size: %0.3fMB" % (n_bytes / 1e6))
        
        projected_dists = euclidean_distances(
            projected_data, squared=True).ravel()[nonzero]
        print('plotting.................................')
        plt.figure()
        min_dist = min(projected_dists.min(), dists.min())
        max_dist = max(projected_dists.max(), dists.max())
        plt.hexbin(dists, projected_dists, gridsize=100, cmap=plt.cm.PuBu,
                   extent=[min_dist, max_dist, min_dist, max_dist])
        plt.xlabel("Pairwise squared distances in original space")
        plt.ylabel("Pairwise squared distances in projected space")
        plt.title("Pairwise distances distribution for n_components=%d" %
                  n_components)
        cb = plt.colorbar()
        cb.set_label('Sample pairs counts')

        rates = projected_dists / dists
        print("Mean distances rate: %0.2f (%0.2f)"
              % (np.mean(rates), np.std(rates)))

        plt.figure()
        plt.hist(rates, bins=50, range=(0., 2.), edgecolor='k', **density_param)
        plt.xlabel("Squared distances rate: projected / original")
        plt.ylabel("Distribution of samples pairs")
        plt.title("Histogram of pairwise distance rates for n_components=%d" %
                  n_components)

        # TODO: compute the expected value of eps and add them to the previous plot
        # as vertical lines / region

        plt.show()
    return rp

