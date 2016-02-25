from unittest import TestCase
import os

from nose.tools import eq_
import pandas as pd

from pyculiarity import detect_ts
from pyculiarity.date_utils import date_format


class TestEdge(TestCase):

    def setUp(self):
        self.path = os.path.dirname(os.path.realpath(__file__))

    def test_check_midnight_date_format(self):
        data = pd.read_csv(os.path.join(self.path,
                                        'midnight_test_data.csv'),
                           usecols=['date', 'value'])

        data.date = date_format(data.date, "%Y-%m-%d %H:%M:%S")
        results = detect_ts(data, max_anoms=0.2, threshold=None,
                            direction='both', plot=False,
                            only_last="day",
                            e_value=True)
        eq_(len(results['anoms'].anoms), len(results['anoms'].expected_value))
