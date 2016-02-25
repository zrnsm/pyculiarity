# -*- coding: utf-8 -*-
__author__ = 'willmcginnis'
from pandas import DataFrame
from numpy import hstack
import statsmodels.api as sm


def stl(data, np=3600):
    """
    Seasonal-Trend decomposition procedure based on LOESS

    data : pandas.Series

    ns : int
        Length of the seasonal smoother.
        The value of  ns should be an odd integer greater than or equal to 3.
        A value ns>6 is recommended. As ns  increases  the  values  of  the
        seasonal component at a given point in the seasonal cycle (e.g., January
        values of a monthly series with  a  yearly cycle) become smoother.

    np : int
        Period of the seasonal component.
        For example, if  the  time series is monthly with a yearly cycle, then
        np=12.
        If no value is given, then the period will be determined from the
        ``data`` timeseries.
    """
    # make sure that data doesn't start or end with nan
    _data = data.copy()
    _data = _data.dropna()

    # here we use the python statsmodels STL decomposition instead of R's
    # decompose

    res = sm.tsa.seasonal_decompose(data.values, model='additive', freq=np)
    res_ts = DataFrame(hstack((res.trend.reshape(-1, 1),
                               res.seasonal.reshape(-1, 1),
                               res.resid.reshape(-1, 1))),
                       index=_data.index,
                       columns=['trend',
                                'seasonal',
                                'remainder'])
    res_ts = res_ts.fillna(0)
    return res_ts
