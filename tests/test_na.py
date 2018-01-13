from nose.tools import eq_, raises
from pyculiarity import detect_ts, detect_vec
from unittest import TestCase
import pandas as pd
import os
import numpy as np


class TestNAs(TestCase):
    def setUp(self):
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.raw_data = pd.read_csv(os.path.join(self.path,
                                                 'raw_data.csv'),
                                    usecols=['timestamp', 'count'])

    def test_handling_of_leading_trailing_nas(self):
        for i in list(range(10)) + [len(self.raw_data) - 1]:
            self.raw_data.at[i, 'count'] = np.nan

        results = detect_ts(self.raw_data, max_anoms=0.02,
                            direction='both', plot=False)
        eq_(len(results['anoms'].columns), 2)
        eq_(len(results['anoms'].iloc[:,1]), 131)

    @raises(ValueError)
    def test_handling_of_middle_nas(self):
        self.raw_data.at[len(self.raw_data) / 2, 'count'] = np.nan
        detect_ts(self.raw_data, max_anoms=0.02, direction='both')
