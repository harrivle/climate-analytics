import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


df = pd.read_csv('./data/test_disasters_temp_state_month.csv')
df['date'] = pd.to_datetime(df['date'])

out_df = None
conf_df = pd.DataFrame(columns=['state', 'tn', 'fp', 'fn', 'tp'])

for state in df['state'].unique():
    model_df = df[df['state'] == state].sort_values('date').reset_index(drop=True)

    model = sm.tsa.SARIMAX(model_df['ave_temp'], order=(2, 0, 3), seasonal_order=(2, 0, 3, 12), initialization='approximate_diffuse').fit()

    model_df['fitted_temp'] = model.fittedvalues
    model_df['error'] = model_df['ave_temp'] - model_df['fitted_temp']
    model_df['mean'] = model_df['error'].rolling(3, min_periods=1).mean()
    model_df['std'] = model_df['error'].rolling(3, min_periods=1).std()

    plt.plot(model_df['ave_temp'][-24:])
    plt.plot(model_df['fitted_temp'][-24:], color='red')
    plt.title(state)
    plt.show()

    model_df['anomaly'] = np.where(model_df['error'] >= model_df['std'], 1, 0)

    conf = confusion_matrix(model_df['disaster_occurrence'], model_df['anomaly']).ravel()
    conf_df.append([state] + list(conf))

    if out_df is None:
        out_df = model_df
    else:
        out_df.append(model_df)

print(conf_df)
out_df.to_csv('./output/arima_state_month.csv', index=False)
conf_df.to_csv('./output/arima_state_month_confusion.csv', index=False)
