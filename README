# pyculiarity
A Python port of Twitter's AnomalyDetection R Package. The original source and examples are available here: https://github.com/twitter/AnomalyDetection.

## Usage

The package currently requires rpy2 in order to use R's stl function. A
working R installation must be available. In the future, we'll move to an
stl implementation that doesn't require R.
One candidate for this is https://github.com/andreas-h/pyloess.

As in Twitter's package, there are two top level functions, one for timeseries data and one for simple vector processing, detect_ts and detect_vec respectively. The first one expects a two-column Pandas DataFrame consisting of timestamps and values. The second expects either a
single-column DataFrame or a Series.

Here's an example of loading Twitter's example data (included in the tests directory) with Pandas and passing it to Pyculiarity for processing.
```python
from pyculiarity import detect_ts
import pandas as pd
twitter_example_data = pd.read_csv('raw_data.csv',
                                    usecols=['timestamp', 'count'])
results = detect_ts(twitter_example_data,
                    max_anoms=0.02,
                    direction='both', only_last='day')
```

## Run the tests
The tests are run with nose as follows:
```
nosetests .
```

## Copyright and License
Python port Copyright 2015 Nicolas Steven Miller
Original R source Copyright 2015 Twitter, Inc and other contributors

Licensed under the GPLv3