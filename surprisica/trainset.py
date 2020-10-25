
from surprise import Trainset
from six import iteritems


class Trainset(Trainset):
    def __init__(self, ur, ir, ic, n_users, n_items, n_ratings, n_contexts, rating_scale,
                 raw2inner_id_users, raw2inner_id_items, raw2inner_id_contexts):

        self.ur = ur
        self.ir = ir
        self.ic = ic
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self.n_contexts = n_contexts
        self.rating_scale = rating_scale
        self._raw2inner_id_users = raw2inner_id_users
        self._raw2inner_id_items = raw2inner_id_items
        self._raw2inner_id_contexts = raw2inner_id_contexts
        self._global_mean = None
        # inner2raw dicts could be built right now (or even before) but they
        # are not always useful so we wait until we need them.
        self._inner2raw_id_users = None
        self._inner2raw_id_items = None
        self._inner2raw_id_contexts = None

    def knows_context(self, cid):
        return cid in self.ic

    def to_inner_cid(self, rcid):
        try:
            return self._raw2inner_id_contexts[rcid]
        except KeyError:
            raise ValueError(f'Context {rcid} is not part of the trainset.')

    def to_raw_cid(self, icid):
        if self._inner2raw_id_contexts is None:
            self._inner2raw_id_contexts = {inner: raw for (raw, inner) in
                                           iteritems(self._raw2inner_id_contexts)}

        try:
            return self._inner2raw_id_contexts[icid]
        except KeyError:
            raise ValueError(f'{icid} is not a valid inner id.')
            
    def all_contexts(self):
        return range(self.n_contexts)
