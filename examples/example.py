from pyculiarity import detect_ts
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import datetime

matplotlib.style.use('ggplot')

__author__ = 'willmcginnis'

if __name__ == '__main__':
    # first run the models
    twitter_example_data = pd.read_csv('../tests/raw_data.csv', usecols=['timestamp', 'count'])
    print(twitter_example_data['timestamp'].values[:10])
    twitter_example_data['timestamp'] = twitter_example_data['timestamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())
    print(twitter_example_data['timestamp'].values[:10])
    results = detect_ts(twitter_example_data, max_anoms=0.05, alpha=0.001, direction='both', verbose=True)
    print(results['anoms']['timestamp'].values[:10])

    # format the twitter data nicely
    twitter_example_data['timestamp'] = pd.to_datetime(twitter_example_data['timestamp'])
    twitter_example_data.set_index('timestamp', drop=True)

    twitter_example_data.to_csv('raw.csv', index=False)
    results['anoms'].to_csv('results.csv', index=False)

    # make a nice plot
    f, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(twitter_example_data['timestamp'], twitter_example_data['count'], 'b')
    ax[0].plot(results['anoms'].index, results['anoms']['anoms'], 'ro')
    ax[0].set_title('Detected Anomalies')
    ax[1].set_xlabel('Time Stamp')
    ax[0].set_ylabel('Count')
    ax[1].plot(results['anoms'].index, results['anoms']['anoms'], 'b')
    ax[1].set_ylabel('Anomaly Magnitude')
    plt.show()

