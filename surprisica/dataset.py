from __future__ import (absolute_import, print_function, unicode_literals, division)

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import minmax_scale
from surprise.dataset import Dataset

from .trainset import Trainset


class Dataset(Dataset):
    """Base class for loading dataset.

    Note that you should never instantiate the :class:`Dataset` class directly
    (same goes for its delivered classes), but instead use one of the two available
    methods for loading data."""

    def __init__(self, reader):
        super(Dataset, self).__init__(reader)
        self.context = reader.context

    @classmethod
    def load_from_df(cls, df, reader):
        """Load from a dataframe.

        Use this if your custom dataset is in pandas dataframe format.

        Args:
            df: The dataframe containing the ratings. It must have three columns,
            corresponding to user (raw) ids, item (raw) ids and timestamps.
            reader(:obj:`Reader<surprisica.reader.Reader>`): A reader to read the file.
        Returns:
            rating profile"""
        ratings = cls.create_profile(df=df, reader=reader)

        return DatasetAutoFolds(df=ratings, reader=reader)

    @classmethod
    def load_from_file(cls, path, reader):
        """Load from a path/file.

        Use this if your custom dataset is in .csv format.

        Args:
            reader(:obj:`Reader<surprisica.reader.Reader>`): A reader to read the file.
        Returns:
            rating profile"""
        ratings = cls.read_ratings(cls, file_name=path, reader=reader)

        return DatasetAutoFolds(df=ratings, reader=reader)

    @classmethod
    def create_profile(cls, df, reader):
        """Create rating profile."""
        if reader.context:
            ratings = cls.create_rating_profile(df, reader)
            contexts = cls.create_context_profile(df, reader)
            profile = ratings.merge(contexts)

        else:
            profile = cls.create_rating_profile(df, reader)

        return profile

    @staticmethod
    def create_rating_profile(df, reader):
        """Generate ratings of each location based the number of visits
        in each location.

        Args:
            df: raw rating in pandas dataframe format.
            reader(:obj:`Reader<surprisica.reader.Reader>`): A reader to read the file.
        Returns:
            A dataframe containing user ids, item ids, rating and number of visits from each item"""
        col = df.columns.to_list()
        uid = col[0]
        iid = col[1]
        t = col[2]
        r = df.groupby([uid, iid])[t].count().reset_index(name='visit')
        r[['rating']] = minmax_scale(r[['visit']], feature_range=reader.rating_scale)
        r = r[[uid, iid, 'rating', 'visit']]

        return r

    @staticmethod
    def create_context_profile(df, reader):
        """Generate the context profile regarding visits.

        Args:
            df: raw rating in pandas dataframe format.
            reader(:obj:`Reader<surprisica.reader.Reader>`): A reader to read the file.
        Returns:
            A dataframe containing item ids, contexts, weights of contexts"""
        col = df.columns.to_list()
        col.remove(col[2])
        lid = col[1]
        cnx = col[2:]
        df = df.filter(col)
        df = df.drop_duplicates()
        # Check context exists
        for c in reader.cnx_entities:
            if df[c].dtype != int:
                raise ValueError('Column "{0}" should have integer type.'.format(c))

        cnx_loc = df.groupby([lid] + cnx)[lid].count()
        all_loc = df.groupby(cnx)[lid].count()
        usr_loc = df.groupby([lid])[lid].count()
        # Term frequency
        tf = cnx_loc.div(all_loc).to_frame(name='tf').reset_index()
        # Inverse document frequency
        idf = np.log10(usr_loc.apply(lambda x: usr_loc.sum() / x)).to_frame(name='idf').reset_index()
        res = tf.merge(idf)
        res['cnx_weight'] = res['tf'] * res['idf']
        res = res.drop(['tf', 'idf'], axis=1)
        res = df.merge(res)

        return res

    def construct_trainset(self, raw_trainset):
        raw2inner_id_users = {}
        raw2inner_id_items = {}
        raw2inner_id_contexts = {} if self.context else None

        current_u_index = 0
        current_i_index = 0
        current_c_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)
        iv = defaultdict(list)
        uc = defaultdict(list) if self.context else None
        ic = defaultdict(list) if self.context else None

        for item in raw_trainset:

            if self.context:
                ruid, riid, r, all_c, dets = item
            else:
                ruid, riid, r, dets = item

            try:
                uid = raw2inner_id_users[ruid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[ruid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[riid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[riid] = current_i_index
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))
            iv[iid].append((uid, dets['visit']))

            if self.context:
                for i in range(len(all_c)):
                    rcid = all_c[i]
                    cw = dets['cw'][i]
                    try:
                        cid = raw2inner_id_contexts[rcid]
                    except KeyError:
                        cid = current_c_index
                        raw2inner_id_contexts[rcid] = current_c_index
                        current_c_index += 1

                    if (iid, cid) not in uc[uid]:
                        uc[uid].append((iid, cid))
                    if (cid, cw) not in ic[iid]:
                        ic[iid].append((cid, cw))

                    uc[uid].sort()
                    ic[iid].sort()

        n_users = len(ur)
        n_items = len(ir)
        n_ratings = len(raw_trainset)
        n_contexts = len(raw2inner_id_contexts) if self.context else None

        trainset = Trainset(ur,
                            ir,
                            uc,
                            ic,
                            iv,
                            n_users,
                            n_items,
                            n_ratings,
                            n_contexts,
                            self.reader.rating_scale,
                            raw2inner_id_users,
                            raw2inner_id_items,
                            raw2inner_id_contexts)
        return trainset

    def read_ratings(self, file_name, reader):
        """Return a list of ratings (user, item, rating, timestamp, ...) read from
        .csv file."""
        ratings = []
        try:
            raw_ratings = pd.read_csv(file_name,
                                      sep=reader.sep,
                                      encoding='utf-8',
                                      usecols=reader.entities)
            raw_ratings = raw_ratings[reader.entities]
            ratings = self.create_profile(raw_ratings, reader)
        except Exception as err:
            print(err)

        return ratings

    def construct_testset(self, raw_testset):
        if not self.context:
            return super(Dataset, self).construct_testset(raw_testset)
        else:
            return [(ruid, riid, r_ui_trans, rcid)
                    for (ruid, riid, r_ui_trans, rcid, _) in raw_testset]


class DatasetUserFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
        cross-validation) are predefined."""

    def __init__(self, folds_files=None, reader=None):

        Dataset.__init__(self, reader)
        self.folds_files = folds_files

        # check that all files actually exist.
        for train_test_files in self.folds_files:
            for f in train_test_files:
                if not os.path.isfile(os.path.expanduser(f)):
                    raise ValueError('File ' + str(f) + ' does not exist.')


class DatasetAutoFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are not predefined. (Or for when there are no folds at
    all)."""

    def __init__(self, df=None, reader=None):
        super(DatasetAutoFolds, self).__init__(reader=reader)

        if df is not None:
            self.df = df
        else:
            raise ValueError('Must specify rating dataframe.')

        self.raw_ratings = []
        if reader.context:
            cols = self.df.columns.to_list()

            for i, group in self.df.groupby(cols[:3]):
                uid = i[0]
                iid = i[1]
                r = i[2]
                v = group.visit.values[0]
                all_c = []
                all_cw = []
                # All not necessary details about records
                dets = {}
                for t in group.itertuples(index=False, name=None):
                    cnum = len(reader.cnx_entities)
                    all_c.append((t[3:3 + cnum]))
                    all_cw.append(float(t[-1]))
                # Input context weights and inverse item frequency of visits
                dets['cw'], dets['visit'] = all_cw, v
                self.raw_ratings.append((uid, iid, float(r), all_c, dets))

        else:
            for uid, iid, r, v in self.df.itertuples(index=False):
                dets = {}
                dets['visit'] = v
                self.raw_ratings.append((uid, iid, float(r), dets))

    def build_full_trainset(self):
        """Do not split the dataset into folds and just return a trainset as
        is, built from the whole dataset.

        Returns:
            A :class:`Trainset<surprisica.Trainset>` class"""
        return self.construct_trainset(self.raw_ratings)
