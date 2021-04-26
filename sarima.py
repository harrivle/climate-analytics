import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


class SARIMA(object):
    def __init__(self):
        self.data = None
        self.out = None
        self.confusion = None
        self.model = None

    def load_data(self, fp, return_df=False):
        self.data = pd.read_csv(fp)
        self.data['date'] = pd.to_datetime(self.data['date'])

        if return_df:
            return self.data

    def train(self, df=None):
        if df is None:
            df = self.data

        self.model = sm.tsa.SARIMAX(df['ave_temp'], order=(2, 0, 3), seasonal_order=(2, 0, 3, 12), initialization='approximate_diffuse').fit()

        df['fitted_temp'] = self.model.fittedvalues
        df['error'] = df['ave_temp'] - df['fitted_temp']
        df['mean'] = df['error'].rolling(3, min_periods=1).mean()
        df['std'] = df['error'].rolling(3, min_periods=1).std()
        df['anomaly'] = np.where(df['error'] >= df['std'] * 1.5, 1, 0)

        self.out = df

    def eval(self):
        conf = confusion_matrix(self.out['disaster_occurrence'], self.out['anomaly']).ravel()
        self.confusion = {'tn': conf[0], 'fp': conf[1], 'fn': conf[2], 'tp': conf[3]}

    def plot(self):
        plt.plot(self.out['ave_temp'][-24:])
        plt.plot(self.out['fitted_temp'][-24:], color='red')
        plt.show()


def main():
    data_fp = './data/test_disasters_temp_state_month.csv'
    out_df = None
    conf_df = pd.DataFrame()

    states = SARIMA().load_data(data_fp, return_df=True)['state'].unique()

    for state in states:
        sarima = SARIMA()
        model_df = sarima.load_data(data_fp, return_df=True)
        model_df = model_df[model_df['state'] == state].sort_values('date').reset_index(drop=True)

        sarima.train(model_df)
        sarima.eval()

        sarima.confusion['state'] = state
        conf_df = conf_df.append(sarima.confusion, ignore_index=True)
        if out_df is None:
            out_df = model_df
        else:
            out_df = out_df.append(model_df)

    print(conf_df)

    out_df.to_csv('./output/arima_state_month.csv', index=False)
    conf_df.to_csv('./output/arima_state_month_confusion.csv', index=False)


if __name__ == '__main__':
    main()
