# pyculiarity
A Python port of Twitter's AnomalyDetection R Package.

## Usage
```bash
python setup.py install
```
The package currently depends on rpy2 exclusively for R's stl function, so a
working R installation must be available. In the future, we'll move to a proper
LOESS implementation callable from Python without having to resort to R.
One candidate for this is https://github.com/andreas-h/pyloess.

As in Twitter's package, there are two top level functions, one for timeseries data (detect_ts) and one for simple vector processing (detect_vs). The first one expects a two-column Pandas DataFrame consisting of timestamps and values. The second expects either a
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
Python Port Copyright 2015 Nicolas Steven Miller
Original R source Copyright 2015 Twitter, Inc and other contributors

Licensed under the GPLv3