
from __future__ import (absolute_import, print_function, unicode_literals, division)

import pyximport; pyximport.install()
from .. import utils as simfunc
from .predictions import Prediction
from .predictions import PredictionImpossible


class AlgoBase:

    def __init__(self, **kwargs):
        self.sim_options = kwargs.get('sim_options', {})

    def fit(self, trainset):
        self.trainset = trainset

        return self

    def predict(self, uid, iid, cid=None, r_ui=None, clip=True, verbose=False):
        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
            if r_ui is None:
                for (i, r) in self.trainset.ur[iuid]:
                    if i == iiid:
                        r_ui = r
        except ValueError:
            iiid = 'UKN__' + str(iid)

        if cid:
            try:
                icid = self.trainset.to_inner_cid(cid)
            except ValueError:
                icid = 'UKN__' + str(cid)

        details = {}
        try:
            est = self.estimate(iuid, iiid, icid) if cid else self.estimate(iuid, iiid)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.default_prediction()
            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, cid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def default_prediction(self):
        return self.trainset.global_mean

    def compute_similarities(self):
        construct_func = {'cosine': simfunc.cosine,
                          'msd': simfunc.msd,
                          'asymmetric_cosine': simfunc.asymmetric_cosine,
                          'asymmetric_msd': simfunc.asymmetric_msd,
                          'user_influence': simfunc.usr_influence_cos}
        name = self.sim_options.get('name', 'cosine').lower()

        n_x, yr, n_y, xr = self.trainset.n_users, self.trainset.ir, self.trainset.n_items, self.trainset.ur
        min_support = self.sim_options.get('min_support', 1)
        args = [n_x, yr, min_support]

        if name == 'asymmetric_cosine' or name == 'asymmetric_msd':
            args.insert(2, xr)
        if name == 'user_influence':
            args.insert(2, n_y)
            args.insert(3, xr)

        try:
            if getattr(self, 'verbose', False):
                print('Computing the {0} similarity matrix ...'.format(name))
            sim = construct_func[name](*args)
            if getattr(self, 'verbose', False):
                print('Done computing similarity matrix.')
            return sim
        except KeyError:
            raise NameError('Wrong sim name {0}. Allowed values are'.format(name) +
                            ', '.join(construct_func.keys()))

    def get_neighbors(self, iid, k=None):
        all_instances = self.trainset.all_users
        others = [(x, self.sim[iid, x]) for x in all_instances() if (x != iid) and (self.sim[iid, x] != 0)]
        others.sort(key=lambda t: t[1], reverse=True)
        k_nearest_neighbors = [j for (j, _) in others[:k]]

        return k_nearest_neighbors
