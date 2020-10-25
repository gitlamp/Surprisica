
from __future__ import (absolute_import, print_function, unicode_literals, division)

import pyximport; pyximport.install()
import heapq
import numpy as np

from .. import utils as simfunc
from .algo_base import AlgoBase
from .predictions import PredictionImpossible


class CF(AlgoBase):
    def __init__(self, k=5, min_k=1, context_aware=False, **kwargs):
        super(CF, self).__init__(context_aware=context_aware, **kwargs)
        self.k = k
        self.min_k = min_k
        self.context_aware = context_aware
        if self.context_aware:
            self.delta = float(self.sim_options.get('delta'))

    def fit(self, trainset):
        super(CF, self).fit(trainset)
        self.yr = self.trainset.ir

        if self.context_aware:
            self.csim = self.compute_cnx_similarities()
            self.xc = self.trainset.uc
            self.yc = self.trainset.ic

        return self

    def estimate(self, u, i, c=None):
        est = self.trainset.global_mean
        neighbors = []

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        if self.context_aware:
            if not self.trainset.knows_context(c):
                raise PredictionImpossible('Context is unknown.')

        for (x2, r) in self.yr[i]:
            if self.context_aware:
                cnx = [x2c for (x2i, x2c) in self.xc[x2] if x2i == i]
                neighbors.append((self.sim[u, x2], x2, r, cnx))
            else:
                neighbors.append((self.sim[u, x2], r))
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # compute weighted average
        sum_sim = num_csim = sum_ratings = actual_k = 0
        actual_csim = []
        for k in k_neighbors:
            if self.context_aware:
                sim, uid, r, cnx = k
                in_cnx = False

                if isinstance(self.delta, float) and (0 <= self.delta <= 1):
                    for ck in cnx:
                        if self.csim[c, ck] > self.delta:
                            in_cnx = True
                            actual_csim.append(ck)
                else:
                    raise ValueError('Wrong sim delta value. ' +
                                     'Delta should be between 0 and 1')

                if in_cnx:
                    num_csim += 1
            else:
                sim, r = k

            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        if self.context_aware:
            est += (sum_ratings / sum_sim) * (num_csim / actual_k)
        else:
            est += sum_ratings / sum_sim

        actual_csim = np.unique([self.trainset.to_raw_cid(c) for c in actual_csim if actual_csim], axis=0).tolist()
        details = {'actual_k': actual_k, 'actual_sim_cnx': actual_csim}
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
