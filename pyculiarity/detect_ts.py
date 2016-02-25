# -*- coding: utf-8 -*-
from collections import namedtuple
import datetime
import copy

from past.builtins import basestring
from pandas import DataFrame, to_datetime
from pandas.lib import Timestamp
import numpy as np

from pyculiarity.date_utils import get_gran
from pyculiarity.detect_anoms import detect_anoms

Direction = namedtuple('Direction', ['one_tail', 'upper_tail'])


def detect_ts(df, max_anoms=0.10, direction='pos', alpha=0.05, threshold=None, e_value=False, longterm=False,
              piecewise_median_period_weeks=2, granularity='day', verbose=False, inplace=True):
    """
    Anomaly Detection Using Seasonal Hybrid ESD Test
    A technique for detecting anomalies in seasonal univariate time series where the input is a
    series of <timestamp, value> pairs.

    Args:

    x: Time series as a two column data frame where the first column consists of the integer UTC Unix
    timestamps and the second column consists of the observations.

    max_anoms: Maximum number of anomalies that S-H-ESD will detect as a percentage of the
    data.

    direction: Directionality of the anomalies to be detected. Options are: ('pos' | 'neg' | 'both').

    alpha: The level of statistical significance with which to accept or reject anomalies.

    only_last: Find and report anomalies only within the last day or hr in the time series. Options: (None | 'day' | 'hr')

    threshold: Only report positive going anoms above the threshold specified. Options are: (None | 'med_max' | 'p95' | 'p99')

    e_value: Add an additional column to the anoms output containing the expected value.

    longterm: Increase anom detection efficacy for time series that are greater than a month.

    See Details below.
    piecewise_median_period_weeks: The piecewise median time window as described in Vallis, Hochenbaum, and Kejariwal
    (2014). Defaults to 2.

    Details


    'longterm' This option should be set when the input time series is longer than a month.
    The option enables the approach described in Vallis, Hochenbaum, and Kejariwal (2014).
    'threshold' Filter all negative anomalies and those anomalies whose magnitude is smaller
    than one of the specified thresholds which include: the median
    of the daily max values (med_max), the 95th percentile of the daily max values (p95), and the
    99th percentile of the daily max values (p99).

    The returned value is a dictionary with the following components:
      anoms: Data frame containing timestamps, values, and optionally expected values.
      plot: A graphical object if plotting was requested by the user. The plot contains
      the estimated anomalies annotated on the input time series
    """

    if not isinstance(df, DataFrame):
        raise ValueError("data must be a single data frame.")
    else:
        if len(df.columns) != 2 or not df.iloc[:, 1].map(np.isreal).all():
            raise ValueError('''data must be a 2 column data.frame, with the first column being a set of timestamps, and
                                the second coloumn being numeric values.''')

        if not (df.dtypes[0].type is np.float64) and not (df.dtypes[0].type is np.int64):
            raise ValueError('''The input timestamp column must be a float or integer of the unix timestamp, not date
                                time columns, date strings or pd.TimeStamp columns.''')

    if not inplace:
        df = copy.deepcopy(df)

    # change the column names in place, rather than copying the entire dataset, but save the headers to replace them.
    orig_header = df.columns.values
    df.rename(columns={df.columns.values[0]: "timestamp", df.columns.values[1]: "value"}, inplace=True)

    # Sanity check all input parameters
    if max_anoms > 0.49:
        length = len(df.value)
        raise ValueError("max_anoms must be less than 50%% of the data points (max_anoms =%f data_points =%s)." % (round(max_anoms * length, 0), length))

    if direction not in ['pos', 'neg', 'both']:
        raise ValueError("direction options are: pos | neg | both.")

    if not (0.01 <= alpha or alpha <= 0.1):
        if verbose:
            import warnings
            warnings.warn("alpha is the statistical signifigance, and is usually between 0.01 and 0.1")

    if threshold not in [None, 'med_max', 'p95', 'p99']:
        raise ValueError("threshold options are: None | med_max | p95 | p99")

    if not isinstance(e_value, bool):
        raise ValueError("e_value must be a boolean")

    if not isinstance(longterm, bool):
        raise ValueError("longterm must be a boolean")

    if piecewise_median_period_weeks < 2:
        raise ValueError(
            "piecewise_median_period_weeks must be at greater than 2 weeks")

    # if the data is daily, then we need to bump the period to weekly to get multiple examples
    gran = granularity
    gran_period = {
        'ms': 60000,
        'sec': 3600,
        'min': 1440,
        'hr': 24,
        'day': 7
    }
    period = gran_period.get(gran)
    if not period:
        raise ValueError('%s granularity detected. This is currently not supported.' % (gran, ))

    # now convert the timestamp column into a proper timestamp
    df['timestamp'] = df['timestamp'].map(lambda x: datetime.datetime.utcfromtimestamp(x))

    num_obs = len(df.value)

    clamp = (1 / float(num_obs))
    if max_anoms < clamp:
        max_anoms = clamp

    if longterm:
        if gran == "day":
            num_obs_in_period = period * piecewise_median_period_weeks + 1
            num_days_in_period = 7 * piecewise_median_period_weeks + 1
        else:
            num_obs_in_period = period * 7 * piecewise_median_period_weeks
            num_days_in_period = 7 * piecewise_median_period_weeks

        last_date = df.timestamp.iget(-1)

        all_data = []

        for j in range(0, len(df.timestamp), num_obs_in_period):
            start_date = df.timestamp.iget(j)
            end_date = min(start_date + datetime.timedelta(days=num_obs_in_period), df.timestamp.iget(-1))

            # if there is at least 14 days left, subset it, otherwise subset last_date - 14days
            if (end_date - start_date).days == num_days_in_period:
                sub_df = df[(df.timestamp >= start_date) & (df.timestamp < end_date)]
            else:
                sub_df = df[(df.timestamp > (last_date - datetime.timedelta(days=num_days_in_period))) & (df.timestamp <= last_date)]
            all_data.append(sub_df)
    else:
        all_data = [df]

    all_anoms = DataFrame(columns=['timestamp', 'value'])
    seasonal_plus_trend = DataFrame(columns=['timestamp', 'value'])

    # Detect anomalies on all data (either entire data in one-pass, or in 2 week blocks if longterm=TRUE)
    for i in range(len(all_data)):
        directions = {
            'pos': Direction(True, True),
            'neg': Direction(True, False),
            'both': Direction(False, True)
        }
        anomaly_direction = directions[direction]

        # detect_anoms actually performs the anomaly detection and returns the result in a list containing the anomalies
        # as well as the decomposed components of the time series for further analysis.

        s_h_esd_timestamps = detect_anoms(all_data[i],
                                          k=max_anoms,
                                          alpha=alpha,
                                          num_obs_per_period=period,
                                          use_decomp=True,
                                          one_tail=anomaly_direction.one_tail,
                                          upper_tail=anomaly_direction.upper_tail,
                                          verbose=verbose)
        if s_h_esd_timestamps is None:
            return {
                'anoms': DataFrame(columns=["timestamp", "anoms"])
            }

        # store decomposed comps in local variable and overwrite s_h_esd_timestamps to contain only the anom timestamps
        data_decomp = s_h_esd_timestamps['stl']
        s_h_esd_timestamps = s_h_esd_timestamps['anoms']

        # -- Step 3: Use detected anomaly timestamps to extract the actual anomalies (timestamp and value) from the data
        if s_h_esd_timestamps:
            anoms = all_data[i][all_data[i].timestamp.isin(s_h_esd_timestamps)]
        else:
            anoms = DataFrame(columns=['timestamp', 'value'])

        # Filter the anomalies using one of the thresholding functions if applicable
        if threshold:
            # Calculate daily max values
            periodic_maxes = df.groupby(df.timestamp.map(Timestamp.date)).aggregate(np.max).value

            # Calculate the threshold set by the user
            thresh = 0.5
            if threshold == 'med_max':
                thresh = periodic_maxes.median()
            elif threshold == 'p95':
                thresh = periodic_maxes.quantile(.95)
            elif threshold == 'p99':
                thresh = periodic_maxes.quantile(.99)

            # Remove any anoms below the threshold
            anoms = anoms[anoms.value >= thresh]

        all_anoms = all_anoms.append(anoms)
        seasonal_plus_trend = seasonal_plus_trend.append(data_decomp)

    # Cleanup potential duplicates
    try:
        all_anoms.drop_duplicates(subset=['timestamp'])
        seasonal_plus_trend.drop_duplicates(subset=['timestamp'])
    except TypeError:
        all_anoms.drop_duplicates(cols=['timestamp'])
        seasonal_plus_trend.drop_duplicates(cols=['timestamp'])

    # Calculate number of anomalies as a percentage
    anom_pct = (len(df.value) / float(num_obs)) * 100

    # name the columns back
    df.rename(columns={"timestamp": orig_header[0], "value": orig_header[1]}, inplace=True)

    if anom_pct == 0:
        return {"anoms": None}

    all_anoms.index = all_anoms.timestamp

    if e_value:
        d = {
            'timestamp': all_anoms.timestamp,
            'anoms': all_anoms.value,
            'expected_value': seasonal_plus_trend[
                seasonal_plus_trend.timestamp.isin(
                    all_anoms.timestamp)].value
        }
    else:
        d = {
            'timestamp': all_anoms.timestamp,
            'anoms': all_anoms.value
        }

    anoms = DataFrame(d, index=d['timestamp'].index)

    # convert timestamps back to unix time
    anoms['timestamp'] = anoms['timestamp'].astype(np.int64)
    anoms['timestamp'] = anoms['timestamp'].map(lambda x: x * 10e-10)

    return {'anoms': anoms}
