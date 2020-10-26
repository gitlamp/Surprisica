
from __future__ import (absolute_import, print_function, unicode_literals, division)

import numpy as np


class Reader:

    def __init__(self, rating_scale=(1, 5), line_format='user location timestamp', sep=None):
        self.sep = sep
        self.rating_scale = rating_scale
        splitted_format = line_format.lower().split()
        entities = ['user', 'location', 'timestamp']
        self.cnx_entities = np.setdiff1d(splitted_format, entities, assume_unique=True).tolist()

        if self.cnx_entities:
            self.context = True
            entities += self.cnx_entities

        for n in entities:
            if n not in splitted_format:
                raise ValueError('line_format parameter is incorrect.')
            else:
                super(Reader, self).__setattr__(n, n)

        self.indexes = [splitted_format.index(entity) for entity in entities]

    def parse_line(self, line):
        line = line.split(self.sep)
        try:
            if self.context:
                uid, lid, r, s, dt, w = (line[i].strip() for i in self.indexes)
                return uid, lid, float(r), s, dt, w
            else:
                uid, lid, r = (line[i].strip() for i in self.indexes)
                return uid, lid, float(r), None
        except IndexError:
            raise ValueError('Impossible to parse line.')
