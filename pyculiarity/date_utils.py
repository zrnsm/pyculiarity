# -*- coding: utf-8 -*-
from datetime import datetime
from heapq import nlargest
from re import match

import pytz
import numpy as np


def date_format(column, format):
    return column.map(lambda datestring: datetime.strptime(datestring, format))


def get_gran(tsdf, index=0):
    col = tsdf.iloc[:, index]

    largest, second_largest = nlargest(2, col)
    gran = int(round(largest - second_largest))

    if gran >= 86400:
        return "day"
    elif gran >= 3600:
        return "hr"
    elif gran >= 60:
        return "min"
    elif gran >= 1:
        return "sec"
    else:
        return "ms"
