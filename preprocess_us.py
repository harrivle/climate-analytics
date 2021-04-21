import re

import numpy as np
import pandas as pd
import datetime as dt
import gc
import random
import sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def main():
    dis_path = './data/us_disaster_declarations.csv'
    temp_path = './data/GlobalLandTemperaturesByState.csv'
    states_path = './data/states.csv'

    # load list of states
    states = {}
    with open(states_path) as f:
        next(f)

        for line in f:
            l = line.split(',')
            states[l[0].strip()] = l[1].strip()

    # print(states)

    # filter disaster dataset
    dis_data = pd.read_csv(dis_path)[['state', 'declaration_date', 'incident_type', 'declaration_title']].rename(
        {'declaration_date': 'date'}, axis=1)
    dis_data['date'] = dis_data['date'].astype('datetime64[ns]').dt.strftime('%m-%Y')
    dis_data = dis_data.drop_duplicates(subset=['incident_type', 'declaration_title', 'date', 'state'], keep='first')
    # dis_data = dis_data.groupby(['state', 'date']).count()
    dis_data['disaster_type'] = dis_data['incident_type']
    dis_data = dis_data.rename({'incident_type': 'disaster_occurrence'}, axis=1)
    dis_data['disaster_occurrence'] = np.ones(dis_data['disaster_occurrence'].shape)
    # dis_data = dis_data.reset_index()

    # print(dis_data)

    dis_data.to_csv('./data/test_disasters_state_month.csv', index=False)

    # filter temperature dataset
    temp_data = pd.read_csv(temp_path)
    temp_data = temp_data[temp_data['Country'] == 'United States'].dropna()  # filter by United States, remove NaNs
    temp_data['date'] = temp_data['dt'].astype('datetime64[ns]').dt.strftime(
        '%m-%Y')  # convert string to date, then convert to year
    temp_data['state'] = temp_data['State'].apply(
        lambda x: states[x] if x in states else None)  # preprocess state strings
    temp_data = temp_data.dropna()
    temp_data = temp_data.groupby(['date', 'state'])  # group by year then state
    temp_data = temp_data[
        ['AverageTemperature', 'AverageTemperatureUncertainty']].mean().reset_index()  # take average over groups

    # print(temp_data)

    temp_data.to_csv('./data/test_temp_state_month.csv', index=False)

    # join on `Year` and `State`
    df = pd.merge(temp_data, dis_data, on=['date', 'state'], how='left').set_index(['date', 'state'], drop=True)
    df.rename({'AverageTemperature': 'ave_temp', 'AverageTemperatureUncertainty': 'ave_temp_uncertainty'}, axis=1,
              inplace=True)
    df = df.fillna(0).reset_index()
    df['month'] = df['date'].astype('datetime64[ns]').dt.strftime('%m')

    df.to_csv('./data/test_disasters_temp_state_month.csv', index=False)

    df['y_data'] = df['disaster_occurrence']
    df = df.drop(['disaster_occurrence'], axis=1)
    print(df)

    x_data = df.iloc[:, 0:-1]
    y_data = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=.7, random_state=614, shuffle=True)

    print(x_data)
    print(y_data)


if __name__ == '__main__':
    main()

