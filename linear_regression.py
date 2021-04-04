import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from scipy import stats
from sklearn import preprocessing


def main():
    fp = './data/yearly_temp_disaster_by_state.csv'

    df = pd.read_csv(fp)

    df = df[df['incident_type'] == 'Drought']
    df = df.groupby(['date', 'state'])[['AverageTemperature', 'incident_type']].agg(['mean', 'count'])
    df.columns = df.columns.droplevel(0)
    df = df.reset_index().rename({'mean': 'temp'}, axis=1)

    print(df)

    corr, _ = stats.pearsonr(df['temp'], df['count'])

    print(corr)

    plt.plot(df['temp'], df['count'], 'o')
    plt.show()


if __name__ == '__main__':
    main()
