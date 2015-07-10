# -*- coding: utf-8 -*-
from collections import namedtuple
from date_utils import format_timestamp, get_gran, date_format, datetimes_from_ts
from detect_anoms import detect_anoms
from math import ceil
from pandas import DataFrame
from pandas.lib import Timestamp
import datetime
import numpy as np

Direction = namedtuple('Direction', ['one_tail', 'upper_tail'])

def detect_ts(df, max_anoms=0.10, direction='pos',
              alpha=0.05, only_last=None, threshold=None,
              e_value=False, longterm=False,
              piecewise_median_period_weeks=2, plot=False,
              y_log=False, xlabel = '', ylabel = 'count',
              title=None, verbose=False):
    if not isinstance(df, DataFrame):
        raise ValueError("data must be a single data frame.")
    else:
        if len(df.columns) != 2 or not df.iloc[:,1].map(np.isreal).all():
            raise ValueError(("data must be a 2 column data.frame, with the"
                              "first column being a set of timestamps, and "
                              "the second coloumn being numeric values."))

        if (not (df.dtypes[0].type is np.datetime64)
            and not (df.dtypes[0].type is np.int64)):
            df = format_timestamp(df)

    if list(df.columns.values) != ["timestamp", "count"]:
        df.columns = ["timestamp", "count"]

    # Sanity check all input parameters
    if max_anoms > 0.49:
        length = len(df.iloc[:,1])
        raise ValueError(
            ("max_anoms must be less than 50% of "
             "the data points (max_anoms =%f data_points =%s).")
                         % (round(max_anoms * length, 0), length))

    if not direction in ['pos', 'neg', 'both']:
        raise ValueError("direction options are: pos | neg | both.")

    if not (0.01 <= alpha or alpha <= 0.1):
        if verbose:
            import warnings
            warnings.warn(("alpha is the statistical signifigance, "
                           "and is usually between 0.01 and 0.1"))

    if only_last and not only_last in ['day', 'hr']:
        raise ValueError("only_last must be either 'day' or 'hr'")

    if not threshold in [None,'med_max','p95','p99']:
        raise ValueError("threshold options are: None | med_max | p95 | p99")

    if not isinstance(e_value, bool):
        raise ValueError("e_value must be a boolean")

    if not isinstance(longterm, bool):
        raise ValueError("longterm must be a boolean")

    if piecewise_median_period_weeks < 2:
        raise ValueError(
            "piecewise_median_period_weeks must be at greater than 2 weeks")

    if not isinstance(plot, bool):
        raise ValueError("plot must be a boolean")

    if not isinstance(y_log, bool):
        raise ValueError("y_log must be a boolean")

    if not isinstance(xlabel, basestring):
        raise ValueError("xlabel must be a string")

    if not isinstance(ylabel, basestring):
        raise ValueError("ylabel must be a string")

    if title and not isinstance(title, basestring):
        raise ValueError("title must be a string")

    if not title:
        title = ''
    else:
        title = title + " : "

    gran = get_gran(df)

    if gran == "day":
        num_days_per_line = 7
        if isinstance(only_last, basestring) and only_last == 'hr':
            only_last = 'day'
    else:
        num_days_per_line = 1

    if gran == 'sec':
        df.timestamp = date_format(df.timestamp, "%Y-%m-%d %H:%M:00")
        df = format_timestamp(df.groupby('timestamp').aggregate(np.sum))

    # if the data is daily, then we need to bump
    # the period to weekly to get multiple examples
    gran_period = {
        'min': 1440,
        'hr': 24,
        'day': 7
    }
    period = gran_period[gran]
    num_obs = len(df['count'])

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

        all_data = range(int(ceil(len(df['count']) / float(num_obs_in_period))))

        for j in range(0, len(df.timestamp), num_obs_in_period):
            start_date = df.timestamp.iget(j)
            end_date = min(start_date
                           + datetime.timedelta(days=num_obs_in_period),
                           df.timestamp.iget(-1))

            # if there is at least 14 days left, subset it,
            # otherwise subset last_date - 14days
            if (end_date - start_date).days == num_days_in_period:
                all_data[int(ceil(j / num_obs_in_period))] = df[
                    (df.timestamp >= start_date) & (df.timestamp < end_date)]
            else:
                all_data[int(ceil(j / num_obs_in_period))] = df[
                    (df.timestamp >
                     (last_date - datetime.timedelta(days=num_days_in_period)))
                    & (df.timestamp <= last_date)]
    else:
        all_data = [df]

    all_anoms = DataFrame(columns=['timestamp', 'count'])
    seasonal_plus_trend = DataFrame(columns=['timestamp', 'count'])

    # Detect anomalies on all data (either entire data in one-pass,
    # or in 2 week blocks if longterm=TRUE)
    for i in range(len(all_data)):
        directions = {
            'pos': Direction(True, True),
            'neg': Direction(True, False),
            'both': Direction(False, True)
        }
        anomaly_direction = directions[direction]

        # detect_anoms actually performs the anomaly detection and
        # returns the results in a list containing the anomalies
        # as well as the decomposed components of the time series
        # for further analysis.

        s_h_esd_timestamps = detect_anoms(all_data[i], k=max_anoms, alpha=alpha,
                                          num_obs_per_period=period,
                                          use_decomp=True, use_esd=False,
                                          one_tail=anomaly_direction.one_tail,
                                          upper_tail=anomaly_direction.upper_tail,
                                          verbose=verbose)

        # store decomposed components in local variable and overwrite
        # s_h_esd_timestamps to contain only the anom timestamps
        data_decomp = s_h_esd_timestamps['stl']
        s_h_esd_timestamps = s_h_esd_timestamps['anoms']

        # -- Step 3: Use detected anomaly timestamps to extract the actual
        # anomalies (timestamp and value) from the data
        if s_h_esd_timestamps:
            anoms = all_data[i][all_data[i].timestamp.isin(s_h_esd_timestamps)]
        else:
            anoms = DataFrame(columns=['timestamp', 'count'])

        # Filter the anomalies using one of the thresholding functions if applicable
        if threshold:
            # Calculate daily max values
            periodic_maxes = df.groupby(
                df.timestamp.map(Timestamp.date)).aggregate(np.max)['count']

            # Calculate the threshold set by the user
            if threshold == 'med_max':
                thresh = periodic_maxes.median()
            elif threshold == 'p95':
                thresh = periodic_maxes.quantile(.95)
            elif threshold == 'p99':
                thresh = periodic_maxes.quantile(.99)

            # Remove any anoms below the threshold
            anoms = anoms[anoms['count'] >= thresh]

        all_anoms = all_anoms.append(anoms)
        seasonal_plus_trend = seasonal_plus_trend.append(data_decomp)

    # Cleanup potential duplicates
    all_anoms.drop_duplicates(subset=['timestamp'])
    seasonal_plus_trend.drop_duplicates(subset=['timestamp'])

    # -- If only_last was set by the user,
    # create subset of the data that represent the most recent day
    if only_last:
        start_date = df.timestamp.iget(-1) - datetime.timedelta(days=7)
        start_anoms = df.timestamp.iget(-1) - datetime.timedelta(days=1)
        if gran is "day":
            breaks = 3 * 12
            num_days_per_line = 7
        else:
            if only_last == 'day':
                breaks = 12
            else:
                start_date = df.timestamp.iget(-1) - datetime.timedelta(days=2)
                # truncate to days
                start_date = datetime.date(start_date.year,
                                           start_date.month, start_date.day)
                start_anoms = (df.timestamp.iget(-1)
                               - datetime.timedelta(hours=1))
                breaks = 3

        # subset the last days worth of data
        x_subset_single_day = df[df.timestamp > start_anoms]
        # When plotting anoms for the last day only
        # we only show the previous weeks data
        x_subset_week = df[(df.timestamp <= start_anoms)
                           & (df.timestamp > start_date)]
        if len(all_anoms) > 0:
            all_anoms = all_anoms[all_anoms.timestamp >=
                                  x_subset_single_day.timestamp.iget(0)]
        num_obs = len(x_subset_single_day['count'])

    # Calculate number of anomalies as a percentage
    anom_pct = (len(df.iloc[:,1]) / float(num_obs)) * 100

    if anom_pct == 0:
        return {
            "anoms": None,
            "plot": None
        }

    # The original R implementation handles plotting here.
    # Plotting is currently not implemented in this version.
    # if plot:
    #     plot_something()

#    print 'wtf'
#    print all_anoms

    if e_value:
        d = {
            'timestamp': all_anoms.timestamp,
            'anoms': all_anoms['count'],
            'expected_value': seasonal_plus_trend.iloc[:,1][
                seasonal_plus_trend.timestamp.isin(all_anoms.timestamp)]
        }
    else:
        d = {
            'timestamp': all_anoms.timestamp,
            'anoms': all_anoms['count']
        }
    anoms = DataFrame(d, index=d['timestamp'].index)

    return {
        'anoms': anoms,
        'plot': None
    }
