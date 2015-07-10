from date_utils import format_timestamp
from itertools import groupby
from math import trunc, sqrt
from r_stl import stl
from scipy.stats import t as student_t
from statsmodels.robust.scale import mad
import numpy as np
import pandas as ps
import statsmodels.api as sm
import sys

def detect_anoms(data, k=0.49, alpha=0.05, num_obs_per_period=None,
                 use_decomp=True, use_esd=False, one_tail=True,
                 upper_tail=True, verbose=False):
    if num_obs_per_period is None:
        raise ValueError("must supply period length for time series decomposition")

    num_obs = len(data)

    # Check to make sure we have at least two periods worth of data for anomaly context
    if num_obs < num_obs_per_period * 2:
        raise ValueError("Anom detection needs at least 2 periods worth of data")

    # Check if our timestamps are posix
    posix_timestamp = data.dtypes[0].type is np.datetime64

    # run length encode result of isnull, check for internal nulls
    if (len(map(lambda x: x[0], list(groupby(ps.isnull(
            ps.concat([ps.Series([np.nan]),
                       data.iloc[:,1],
                       ps.Series([np.nan])])))))) > 3):
        raise ValueError("Data contains non-leading NAs. We suggest replacing NAs with interpolated values (see na.approx in Zoo package).")
    else:
        data = data.dropna()

    # -- Step 1: Decompose data. This returns a univarite remainder which will be used for anomaly detection. Optionally, we might NOT decompose.

    data = data.set_index('timestamp')

    if not isinstance(data.index, ps.Int64Index):
        resample_period = {
            1440: 'T',
            24: 'H',
            7: 'D'
        }
        data = data.resample(resample_period[num_obs_per_period])


    decomp = stl(data['count'], "periodic", np=num_obs_per_period)

    # Remove the seasonal component, and the median of the data to create the univariate remainder
    d = {
        'timestamp': data.index,
        'count': data['count'] - decomp['seasonal'] - data['count'].median()
    }
    data = ps.DataFrame(d)

    p = {
        'timestamp': decomp.index,
        'count': (decomp['trend'] + decomp['seasonal']).truncate().convert_objects(convert_numeric=True)
    }
    data_decomp = ps.DataFrame(p)

    # Maximum number of outliers that S-H-ESD can detect (e.g. 49% of data)
    max_outliers = int(num_obs * k)

    if max_outliers == 0:
        raise ValueError("With longterm=TRUE, AnomalyDetection splits the data into 2 week periods by default. You have %d observations in a period, which is too few. Set a higher piecewise_median_period_weeks." % num_obs)

    ## Define values and vectors.
    n = len(data.iloc[:,0])
    R_idx = range(max_outliers)

    num_anoms = 0

    # Compute test statistic until r=max_outliers values have been
    # removed from the sample.
    for i in range(1, max_outliers + 1):
        if one_tail:
            if upper_tail:
                ares = data['count'] - data['count'].median()
            else:
                ares = data['count'].median() - data['count']
        else:
            ares = (data['count'] - data['count'].median()).abs()

        # protect against constant time series
        data_sigma = mad(data['count'])
        if data_sigma == 0:
            break

        ares = ares / float(data_sigma)

        R = ares.max()

        temp_max_idx = ares[ares == R].index.tolist()[0]

        R_idx[i - 1] = temp_max_idx

        data = data[data.index != R_idx[i - 1]]

        if one_tail:
            p = 1 - alpha / float(n - i + 1)
        else:
            p = 1 - alpha / float(2 * (n - i + 1))

        t = student_t.ppf(p, (n - i - 1))
        lam = t * (n - i) / float(sqrt((n - i - 1 + t**2) * (n - i + 1)))

        if R > lam:
            num_anoms = i

    if num_anoms > 0:
        R_idx = R_idx[:num_anoms]
    else:
        R_idx = None

    return {
        'anoms': R_idx,
        'stl': data_decomp
    }
