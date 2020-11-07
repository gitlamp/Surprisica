
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
        if isinstance(cid, list):
            pred = self.multiple_predict(uid, iid, cid, r_ui, clip, verbose)

        else:
            pred = self.single_predict(uid, iid, cid, r_ui, clip, verbose)

        if verbose and isinstance(pred, list):
            [print(t) for t in pred]
        if verbose and isinstance(pred, tuple):
            print(pred)

        return pred

    def default_prediction(self):
        return self.trainset.global_mean

    def test(self, testset, verbose=False):
        predictions = []
        for pairs in testset:
            if len(pairs) > 3:
                uid, iid, r_ui_trans, cid = pairs
                predictions.append(self.predict(uid, iid, cid, r_ui_trans, verbose=verbose))
            else:
                uid, iid, r_ui_trans = pairs
                predictions.append(self.predict(uid, iid, None, r_ui_trans, verbose=verbose))

        return predictions

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

    def single_predict(self, uid, iid, cid=None, r_ui=None, clip=True, verbose=False):
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
            # Check context if single-element list
            if isinstance(cid, list):
                cid = cid[0]
            try:
                icid = self.trainset.to_inner_cid(cid)
            except ValueError:
                icid = 'UKN__' + str(cid)

        details = {}
        try:
            est = self.estimate(iuid, iiid, icid) if cid else self.estimate(iuid, iiid)
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

        return pred

    def multiple_predict(self, uid, iid, cid=None, r_ui=None, clip=True, verbose=False):
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

        icid_group = []
        for c in cid:
            try:
                icid_group.append(self.trainset.to_inner_cid(c))
            except ValueError:
                icid_group.append('UKN__' + str(c))

        all_est = []
        all_details = []
        for c in icid_group:
            try:
                est = self.estimate(iuid, iiid, c)
                all_est.append(est[0])
                details = est[1]
                details['was_impossible'] = False
                all_details.append(details)

            except PredictionImpossible as e:
                all_est.append(self.default_prediction())
                all_details.append({'was_impossible': True,
                                    'reason': str(e)})

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            all_est = [min(higher_bound, est) for est in all_est]
            all_est = [max(lower_bound, est) for est in all_est]

        cid_group = []
        for icid in icid_group:
            try:
                cid_group.append(self.trainset.to_raw_cid(icid))
            except ValueError:
                cid_group.append(icid)
        pred = [Prediction(uid, iid, c, r_ui, e, d)
                for (c, e, d) in list(zip(cid_group, all_est, all_details))]

        return pred
