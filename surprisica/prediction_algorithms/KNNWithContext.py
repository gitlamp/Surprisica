
from __future__ import (absolute_import, print_function, unicode_literals, division)

import pyximport; pyximport.install()
import heapq
import numpy as np

from .. import utils as simfunc
from .KNN import KNNWithMeans
from .predictions import PredictionImpossible


class CSR(KNNWithMeans):
    def __init__(self, k=5, min_k=1, verbose=False, **kwargs):
        super(CSR, self).__init__(k, min_k, verbose, **kwargs)

    def fit(self, trainset):
        super(CSR, self).fit(trainset)
        self.csim = self.compute_cnx_similarities()
        self.xc = self.trainset.uc
        self.yc = self.trainset.ic

        return self

    def estimate(self, u, i, c):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        if not self.trainset.knows_context(c):
            raise PredictionImpossible('Context is unknown.')

        neighbors = []

        for (x2, r) in self.yr[i]:
            cnx = [xc for (xi, xc) in self.xc[x2] if xi == i]
            neighbors.append((x2, self.sim[u, x2], r, cnx))

        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])
        est = self.means[u]

        # compute weighted average
        sum_sim = num_csim = sum_ratings = actual_k = 0
        for k in k_neighbors:
            uid, sim, r, cnx = k

            for ck in cnx:
                if c == ck:
                    num_csim += 1

            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[uid])
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += (sum_ratings / sum_sim) * (num_csim / actual_k)
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        return est, details

    def compute_cnx_similarities(self):
        n_c, yc = self.trainset.n_contexts, self.trainset.ic
        min_support = self.sim_options.get('min_support', 1)

        if getattr(self, 'verbose', False):
            print('Computing the context similarity ...')
        sim = simfunc.cosine(n_c, yc, min_support)
        if getattr(self, 'verbose', False):
            print('Done computing the context similarity.')

        return sim


class EACA_Post(CSR):

    def __init__(self, k=5, min_k=1, verbose=False, **kwargs):
        super(EACA_Post, self).__init__(k, min_k, verbose, **kwargs)
        self.delta = float(self.sim_options.get('delta'))

    def fit(self, trainset):
        super(EACA_Post, self).fit(trainset)

    def estimate(self, u, i, c):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        if not self.trainset.knows_context(c):
            raise PredictionImpossible('Context is unknown.')
        if not (isinstance(self.delta, float) and (0 <= self.delta <= 1)):
            raise ValueError('Wrong sim delta value. Delta should be between 0 and 1')

        neighbors = []

        for (x2, r) in self.yr[i]:
            cnx = [xc for (xi, xc) in self.xc[x2] if xi == i]
            neighbors.append((x2, self.sim[u, x2], r, cnx))

        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])
        est = self.means[u]

        # compute weighted average
        sum_sim = num_csim = sum_ratings = actual_k = 0
        actual_ck = []
        for k in k_neighbors:
            uid, sim, r, cnx = k
            in_cnx = False
            for ck in cnx:
                if self.csim[c, ck] > self.delta:
                    in_cnx = True
                    actual_ck.append((uid, ck))
            if in_cnx:
                num_csim += 1

            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est += (sum_ratings / sum_sim) * (num_csim / actual_k)

        actual_sim_cnx = [(self.trainset.to_raw_uid(u), self.trainset.to_raw_cid(c)) for u, c in actual_ck if actual_ck]
        details = {'actual_k': actual_k, 'actual_sim_cnx': actual_sim_cnx}
        return est, details
