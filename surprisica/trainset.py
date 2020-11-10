
from __future__ import (absolute_import, print_function, unicode_literals, division)

from surprise import Trainset
from six import iteritems


class Trainset(Trainset):
    def __init__(self, ur, ir, uc, ic, iv, n_users, n_items, n_ratings, n_contexts, rating_scale,
                 raw2inner_id_users, raw2inner_id_items, raw2inner_id_contexts):

        super(Trainset, self).__init__(ur, ir, n_users, n_items, n_ratings, rating_scale,
                                       raw2inner_id_users, raw2inner_id_items)
        self.uc = uc
        self.ic = ic
        self.iv = iv
        self.n_contexts = n_contexts
        self._raw2inner_id_contexts = raw2inner_id_contexts
        self._inner2raw_id_contexts = None

    def knows_context(self, cid):
        return cid in self.ic

    def to_inner_cid(self, rcid):
        try:
            return self._raw2inner_id_contexts[rcid]
        except KeyError:
            raise ValueError('Context {0} is not part of the trainset.'.format(rcid))

    def to_raw_cid(self, icid):
        if self._inner2raw_id_contexts is None:
            self._inner2raw_id_contexts = {inner: raw for (raw, inner) in
                                           iteritems(self._raw2inner_id_contexts)}

        try:
            return self._inner2raw_id_contexts[icid]
        except KeyError:
            raise ValueError('{0} is not a valid inner id.'.format(icid))
            
    def all_contexts(self):
        return range(self.n_contexts)
    
    def build_testset(self):
        return [(self.to_raw_uid(u), self.to_raw_iid(i), r, self.to_raw_cid(c))
                for (u, i, r, c) in self.all_ratings()]
