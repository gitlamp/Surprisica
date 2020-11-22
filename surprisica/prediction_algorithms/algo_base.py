
from __future__ import (absolute_import, print_function, unicode_literals, division)

from .. import utils as simfunc
from .predictions import Prediction
from .predictions import PredictionImpossible


class AlgoBase:
    """Abstract class where is defined the basic behavior of prediction algorithms.

    Args:
        **kwargs: Options related to each prediction algorithms including ``sim_options``"""

    def __init__(self, **kwargs):
        self.sim_options = kwargs.get('sim_options', {})

    def fit(self, trainset):
        """Train an algorithm on a training set.

        This method is used by every fit method of prediction algorithms."""
        self.trainset = trainset

        return self

    def predict(self, uid, iid, cid=None, r_ui=None, clip=True, verbose=False):
        """Generates estimated rating for given user ; item ; [context].

        The ``predict`` method converts all ids for user, item and context, and then call
        ``estimate`` method of each derived prediction algorithms. If prediction is impossible
        the estimation is equal to the default prediction.

        Args:
            uid: The raw user id.
            iid: The raw item id.
            cid(tuple): The raw context id. Default is None.
            r_ui(float): The true rating :math:`r_{ui}`.
            clip: Whether to clip the estimation into the rating scale.
                For example, if :math:`\\hat{r}_{ui}` is :math:`5.5` while the
                rating scale is :math:`[1, 5]`, then :math:`\\hat{r}_{ui}` is
                set to :math:`5`. Same goes if :math:`\\hat{r}_{ui} < 1`.
                Default is ``True``.
            verbose(bool): Whether to print details of the prediction. Default
                is False."""
        if isinstance(cid, list):
            pred = self.multiple_predict(uid, iid, cid, r_ui, clip)

        else:
            pred = self.single_predict(uid, iid, cid, r_ui, clip)

        if verbose and isinstance(pred, list):
            [print(t) for t in pred]
        if verbose and isinstance(pred, tuple):
            print(pred)

        return pred

    def default_prediction(self):
        """Estimates ratings when ``PredictionImpossible`` exception is raised.

        Returns:
            The means of all ratings."""
        return self.trainset.global_mean

    def test(self, testset, verbose=False):
        """Estimates ratings on given testset.

        Args:
            testset: A testset returned by :func:`cross-validation
                <surprisica.model_selection.validation.cross_validation>` or
                by :meth:`build_testset()<surprisica.trainset.Trainset.build_testset>`
            verbose(bool): Whether to print details of the prediction.  Default
                is False."""
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
        """Compute user similarity.

        The way that similarity matrix is generated depends on similarity options
        determined in ``sim_options`` parameters passed as a keyword at the creation of
        each algorithm.

        Returns:
            Similarity matrix."""
        construct_func = {'cosine': {'func': simfunc.cosine},
                          'msd': {'func': simfunc.msd},
                          'asymmetric_cosine': {'func': simfunc.asymmetric_cosine},
                          'asymmetric_msd': {'func': simfunc.asymmetric_msd},
                          'sorensen_idf': {'func': simfunc.sorensen_idf},
                          'user_influence': {'func': simfunc.usr_influence_cos}}
        name = self.sim_options.get('name', 'cosine').lower()

        n_x, yr, n_y, xr, yv = self.trainset.n_users,\
                                    self.trainset.ir,\
                                    self.trainset.n_items,\
                                    self.trainset.ur,\
                                    self.trainset.iv
        min_support = self.sim_options.get('min_support', 1)
        construct_func['cosine']['param'] = construct_func['msd']['param'] = [n_x, yr, min_support]
        construct_func['asymmetric_cosine']['param'] = construct_func['asymmetric_msd']['param'] = [n_x, yr, xr, min_support]
        construct_func['sorensen_idf']['param'] = [n_x, yv, n_y]
        construct_func['user_influence']['param'] = [n_x, yr, n_y, xr, min_support]

        for sim_name in construct_func:
            if name == sim_name:
                args = construct_func[sim_name]['param']

        try:
            if getattr(self, 'verbose', False):
                print('Computing the {0} similarity matrix ...'.format(name))
            sim = construct_func[name]['func'](*args)
            if getattr(self, 'verbose', False):
                print('Done computing similarity matrix.')
            return sim
        except KeyError:
            raise NameError('Wrong sim name {0}. Allowed values are'.format(name) +
                            ', '.join(construct_func.keys()))

    def get_neighbors(self, uid, k=None):
        """Return k nearest neighborhood of any ``uid``.

        Args:
            uid: User inner id.
            k(int): Number of neighbors."""
        all_instances = self.trainset.all_users
        others = [(x, self.sim[uid, x]) for x in all_instances() if (x != uid) and (self.sim[uid, x] != 0)]
        others.sort(key=lambda t: t[1], reverse=True)
        k_nearest_neighbors = [j for (j, _) in others[:k]]

        return k_nearest_neighbors

    def single_predict(self, uid, iid, cid=None, r_ui=None, clip=True):
        """Generates estimated rating when there is only a context for given user or item."""
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

    def multiple_predict(self, uid, iid, cid=None, r_ui=None, clip=True):
        """Generates estimated rating when there is multiple contexts for given user or item."""
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
