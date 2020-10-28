
from __future__ import (absolute_import, print_function, unicode_literals, division)

import pyximport; pyximport.install()
import heapq
import numpy as np
from six import iteritems

from .algo_base import AlgoBase
from .predictions import PredictionImpossible


class KNNBasic(AlgoBase):

    def __init__(self, k=5, min_k=1, verbose=False, **kwargs):
        super(KNNBasic, self).__init__(**kwargs)
        self.k = k
        self.min_k = min_k
        self.verbose = verbose

    def fit(self, trainset):
        super(KNNBasic, self).fit(trainset)
        self.sim = self.compute_similarities()
        self.yr = self.trainset.ir

        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        neighbors = [(self.sim[u, x2], r) for (x2, r) in self.yr[i]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


class KNNWithMeans(KNNBasic):
    def __init__(self, k=5, min_k=1, verbose=False, **kwargs):
        super(KNNWithMeans, self).__init__(k, min_k, verbose, **kwargs)

    def fit(self, trainset):
        super(KNNWithMeans, self).fit(trainset)
        self.n_x = self.trainset.n_users
        self.xr = self.trainset.ur
        self.means = np.zeros(self.n_x)

        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        neighbors = [(x2, self.sim[u, x2], r) for (x2, r) in self.yr[i]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        est = self.means[u]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb])
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        return est, details
