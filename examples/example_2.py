from pyculiarity import detect_ts
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import datetime

matplotlib.style.use('ggplot')

__author__ = 'willmcginnis'

if __name__ == '__main__':
    # first run the models
    example_data = pd.read_csv('db_test_data.csv', usecols=['time_stamp', 'temp'])

    results = detect_ts(example_data, max_anoms=0.05, alpha=0.001, granularity='day', direction='both')

    # format the twitter data nicely
    example_data['time_stamp'] = pd.to_datetime(example_data['time_stamp'])
    example_data.set_index('time_stamp', drop=True)

    # make a nice plot
    f, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(example_data['time_stamp'], example_data['temp'], 'b')
    ax[0].plot(results['anoms'].index, results['anoms']['anoms'], 'ro')
    ax[0].set_title('Detected Anomalies')
    ax[1].set_xlabel('Time Stamp')
    ax[0].set_ylabel('Count')
    ax[1].plot(results['anoms'].index, results['anoms']['anoms'], 'b')
    ax[1].set_ylabel('Anomaly Magnitude')
    plt.show()

