from __future__ import (absolute_import, print_function, unicode_literals, division)

from surprise import Trainset
from six import iteritems


class Trainset(Trainset):
    """A trainset contains all handy data from ratings profile.

    It is used the :meth:`fit()<surprisica.prediction_algorithms.algo_base.AlgoBase.fit>`
    method of every prediction algorithm. You should not try to build such an object by your own.
    It is automatically created by
    :meth:`build_full_trainset()<surprisica.dataset.DatasetAutoFolds.build_full_trainset>` method.

    Args:
        ur(:obj:`defaultdict` of :obj:`list`): The user ratings. A dictionary containing lists of tuples of the form ``(item_inner_id, rating)``.
        ir(:obj:`defaultdict` of :obj:`list`): The item ratings. A dictionary containing lists of tuples of the form ``(user_inner_id, rating)``.
        uc(:obj:`defaultdict` of :obj:`list`): The context profile of users. A dictionary containing lists of tuples of the form ``(item_inner_id, context)``.
        ic(:obj:`defaultdict` of :obj:`list`): The context profile of items. A dictionary containing lists of tuples of the form ``(user_inner_id, context)``.
        iv(:obj:`defaultdict` of :obj:`list`): Visits number from each item.
        n_users: Total number of users.
        n_items: Total number of items.
        n_ratings: Total number of ratings.
        n_contexts: Total number of contexts.
        rating_scale(tuple): The min and max rating of the rating scale.
        raw2inner_id_users: A dictionary containing user raw and inner ids.
        raw2inner_id_items: A dictionary containing item raw and inner ids.
        raw2inner_id_contexts: A dictionary containing context raw and inner ids."""

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
        """Indicates if the context is part of trainset.

        Args:
            cid(int): The inner context id.
        Returns:
            ``True`` if user is part of trainset, else ``False``."""
        return cid in self.ic

    def to_inner_cid(self, rcid):
        """Convert raw context id to inner context id.

        Args:
            rcid(tuple): Raw context id.
        Returns:
            Inner context id."""
        try:
            return self._raw2inner_id_contexts[rcid]
        except KeyError:
            raise ValueError('Context {0} is not part of the trainset.'.format(rcid))

    def to_raw_cid(self, icid):
        """Convert inner context id to raw context id.

        Args:
            rcid(int): Inner context id.
        Returns:
            Raw context id."""
        if self._inner2raw_id_contexts is None:
            self._inner2raw_id_contexts = {inner: raw for (raw, inner) in
                                           iteritems(self._raw2inner_id_contexts)}

        try:
            return self._inner2raw_id_contexts[icid]
        except KeyError:
            raise ValueError('{0} is not a valid inner id.'.format(icid))

    def all_contexts(self):
        """Return total number of contexts."""
        return range(self.n_contexts)

    def build_testset(self):
        """Return a list of ratings that can be used as a testset in the
        :meth:`test()<surprisica.prediction_algorithms.algo_base.AlgoBase.test>`
        method.
        The ratings are all the ratings that are in the trainset, i.e. all the
        ratings returned by the :meth:`all_ratings()<surprisica.Trainset.all_ratings>` generator. This is useful in
        cases where you want to to test your algorithm on the trainset."""
        return [(self.to_raw_uid(u), self.to_raw_iid(i), r, self.to_raw_cid(c))
                for (u, i, r, c) in self.all_ratings()]
