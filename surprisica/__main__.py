
from __future__ import (absolute_import, print_function, unicode_literals, division)

import sys
import argparse
import random
import numpy as np

from surprisica import Dataset
from surprisica.prediction_algorithms import KNNBasic
from surprisica.prediction_algorithms import KNNWithMeans
from surprisica.prediction_algorithms import CSR
from surprisica.prediction_algorithms import EACA_Post
from surprisica.model_selection import cross_validate
from surprisica.model_selection import KFold
from surprisica import __version__


def main():
    class Parser(argparse.ArgumentParser):
        def error(self, msg):
            sys.stderr.write('error: %s\n' % msg)
            self.print_help()
            sys.exit(2)

    parser = Parser(
        description='Check the performance of a prediction algorithm on a dataset. ' +
        'You can choose to automatically split the dataset into folds, ' +
        'or manually specify train and test sets.',
        epilog="""Example: 
        surprisica -algo CSR -params "{'name': 'asymmetric_msd', 'verbose': True}"
        """)

    algo_choices = {
        'KNNBasic': KNNBasic,
        'KNNWithMeans': KNNWithMeans,
        'CSR': CSR,
        'EACA_Post': EACA_Post
    }

    parser.add_argument('-algo', type=str, choices=algo_choices,
                        help='Allowed values are: ' +
                        ', '.join(algo_choices.keys()) + '.',
                        metavar='<prediction algorithm>')
    
    parser.add_argument('-params', type=str, default='{}',
                        help='A dictionary contains all parameters for prediction algorithm. ' +
                             'Example: "{\'name\': \'cosine\', \'delta\': 0.5}"',
                        metavar='<algorithm params>')
    
    parser.add_argument('-reader', type=str, default=None,
                        help='A reader to read the columns of your dataset. ' +
                             'Example for non-contextual data:\n' +
                             '"Reader(line_format=\'user location timestamp\', sep=\'\\t\')" ' +
                             'Example for contextual data: ' +
                             '"Reader(line_format=\'user location timestamp context1 content2 ...\')"',
                        metavar='<reader>')

    parser.add_argument('-load-data', type=str, dest='load_data',
                        default=None,
                        help='A file path to dataset to use.',
                        metavar='<file path>')

    parser.add_argument('-folds-files', type=str, dest='folds_files',
                        default=None,
                        help='A list of custom train and test files. ' +
                             'Ignored if -load-data is set. ' +
                             'The -reader parameter needs to be set.',
                        metavar='<train1 test1 train2 test2 ...>')

    parser.add_argument('-n-folds', type=int, dest='n_folds',
                        default=5,
                        help='The number of folds for cross-validation. ' +
                             'Default is 5.',
                        metavar="<number of folds>")
    
    parser.add_argument('-seed', type=int, default=None,
                        help='The seed to use for random number',
                        metavar='<random seed>')
    
    parser.add_argument('-v', '--version', action='version',
                        version=__version__)

    args = parser.parse_args()

    # setup random number generator
    random.seed(args.seed)
    np.random.seed(args.seed)

    # setup algorithms
    params = eval(args.params)
    if args.algo is None:
        parser.error('No algorithm was specified.')
    algo = algo_choices[args.algo](**params)

    # setup dataset
    if args.load_data is not None:
        if args.reader is None:
            parser.error('-reader parameter is needed.')
        reader = eval(args.reader)
        data = Dataset.load_from_file(args.load_data, reader=reader)
        cv = KFold(n_splits=args.n_folds, random_state=args.seed)
        
    cross_validate(algo, data, cv=cv, verbose=True)


if __name__ == '__main__':
    main()
