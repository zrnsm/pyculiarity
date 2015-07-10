# -*- coding: utf-8 -*-
from collections import namedtuple
from date_utils import format_timestamp, get_gran, date_format, datetimes_from_ts
from detect_anoms import detect_anoms
from math import ceil
from pandas import DataFrame, Series, to_datetime
from pandas.lib import Timestamp
import numpy as np

Direction = namedtuple('Direction', ['one_tail', 'upper_tail'])

def detect_vec(df, max_anoms=0.10, direction='pos',
               alpha=0.05, period=None, only_last=False,
               threshold=None, e_value=False, longterm_period=None,
               plot=False, y_log=False, xlabel='', ylabel='count',
               title=None, verbose=False):

    if (isinstance(df, DataFrame) and
        len(df.columns) == 1 and
        df.iloc[:,0].applymap(np.isreal).all(1)):
        d = {
            'timestamp': range(len(df.iloc[:,0])),
            'count': df.iloc[:,0]
        }
        df = DataFrame(d, index=d['timestamp'])
    elif isinstance(df, Series):
        d = {
            'timestamp': range(len(df)),
            'count': df
        }
        df = DataFrame(d, index=d['timestamp'])
    else:
        raise ValueError(("data must be a single data frame, "
                          "list, or vector that holds numeric values."))

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

    if not period:
        raise ValueError(("Period must be set to the number "
                          "of data points in a single period"))

    if not isinstance(only_last, bool):
        raise ValueError("only_last must be a boolean")

    if not threshold in [None,'med_max','p95','p99']:
        raise ValueError("threshold options are: None | med_max | p95 | p99")

    if not isinstance(e_value, bool):
        raise ValueError("e_value must be a boolean")

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

      # -- Main analysis: Perform S-H-ESD

    num_obs = len(df['count'])

    clamp = (1 / float(num_obs))
    if max_anoms < clamp:
        max_anoms = clamp

      # -- Setup for longterm time series

    # If longterm is enabled, break the data into subset
    # data frames and store in all_data,

    if longterm_period:
        all_data = range(int(ceil(len(df['count']) / float(longterm_period))))
        for j in range(0, len(df.timestamp), longterm_period):
            start_index = df.timestamp.iget(j)
            end_index = min((start_index + longterm_period), num_obs)
            if (end_index - start_index) == longterm_period:
                all_data[int(ceil(j / float(longterm_period)))] = df[
                    (df.timestamp >= start_index) & (df.timestamp <= end_index)]
            else:
                all_data[int(ceil(j / float(longterm_period)))] = df[
                    (df.timestamp >= (num_obs - longterm_period)) &
                    (df.timestamp <= num_obs)]
    else:
        all_data = [df]

    # Create empty data frames to store all anoms and
    # seasonal+trend component from decomposition
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

        s_h_esd_timestamps = detect_anoms(all_data[i], k=max_anoms,
                                          alpha=alpha,
                                          num_obs_per_period=period,
                                          use_decomp=True, use_esd=False,
                                          one_tail=anomaly_direction.one_tail,
                                          upper_tail=anomaly_direction.upper_tail,
                                          verbose=verbose)

        # store decomposed components in local variable and
        # overwrite s_h_esd_timestamps to contain only the anom timestamps
        data_decomp = s_h_esd_timestamps['stl']
        s_h_esd_timestamps = s_h_esd_timestamps['anoms']

        # -- Step 3: Use detected anomaly timestamps to
        # extract the actual anomalies (timestamp and value) from the data
        if s_h_esd_timestamps:
            anoms = all_data[i][all_data[i].timestamp.isin(s_h_esd_timestamps)]
        else:
            anoms = DataFrame(columns=['timestamp', 'count'])


        # Filter the anomalies using one of the thresholding
        # functions if applicable
        if threshold:
            # Calculate daily max values
            if isinstance(all_data[i].index[0], np.int64):
                group = all_data[i].timestamp.map(lambda t: t / period)
            else:
                group = all_data[i].timestamp.map(Timestamp.date)
            periodic_maxes = df.groupby(group).aggregate(np.max)['count']

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


    # -- If only_last was set by the user, create subset of
    # the data that represent the most recent period
    if only_last:
        d = {
            'timestamp': df.timestamp.iloc[-period:],
            'count': df['count'].iloc[-period:]
        }
        x_subset_single_period = DataFrame(d, index = d['timestamp'])
        past_obs = period * 7
        if num_obs < past_obs:
            past_obs = num_obs - period
        # When plotting anoms for the last period only we only show
        # the previous 7 periods of data
        d = {
            'timestamp': df.iloc[:,0].iloc[-past_obs:-period],
            'count': df['count'].iloc[-past_obs:-period]
        }
        x_subset_previous = DataFrame(d, index=d['timestamp'])
        all_anoms = all_anoms[all_anoms.timestamp
                              >= x_subset_single_period.timestamp.iget(0)]
        num_obs = len(x_subset_single_period['count'])

    # Calculate number of anomalies as a percentage
    anom_pct = (len(df.iloc[:,1]) / float(num_obs)) * 100

    if anom_pct == 0:
        return {
            "anoms": None,
            "plot": None
        }

    # The original R implementation handles plotting here.
    # Plotting is currently not implemented.
    # if plot:
    #     plot_something()

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
