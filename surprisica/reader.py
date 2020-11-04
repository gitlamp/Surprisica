
from __future__ import (absolute_import, print_function, unicode_literals, division)

import numpy as np


class Reader:

    def __init__(self, rating_scale=(1, 5), line_format='user location timestamp', sep=None):
        self.sep = sep
        self.rating_scale = rating_scale
        splitted_format = line_format.lower().split()
        self.entities = ['user', 'location', 'timestamp']
        self.cnx_entities = np.setdiff1d(splitted_format, self.entities, assume_unique=True).tolist()

        if self.cnx_entities:
            self.context = True
            self.entities += self.cnx_entities

        for n in self.entities:
            if n not in splitted_format:
                raise ValueError('line_format parameter is incorrect.')
            else:
                super(Reader, self).__setattr__(n, n)

        self.indexes = [splitted_format.index(entity) for entity in self.entities]
