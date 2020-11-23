
from __future__ import (absolute_import, print_function, unicode_literals, division)

import numpy as np


class Reader:
    """The Reader class is used to parse a file containing ratings.

    Such a file is assumed to specify only one rating per line, and each line
    needs to respect the following structure: ::
        user ; location/item ; timestamp ; [context/contexts]
    where the order of the fields and the separator (here ';') may be
    arbitrarily defined. Brackets indicate that the timestamp
    field is optional.

    Args:
        rating_scale(tuple): The scale of ratings used for each rating. Default is ``(1,5)``
        line_format(str): The fields name. Please note that ``line_format`` is always space-separated.
        sep(str): The separator between fields. Default is ``','``"""

    def __init__(self, rating_scale=(1, 5), line_format='user location timestamp', sep=','):
        self.sep = sep
        self.rating_scale = rating_scale
        self.context = False
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
