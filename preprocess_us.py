import re

import numpy as np
import pandas as pd


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
    dis_data = pd.read_csv(dis_path)[['state', 'declaration_date', 'incident_type']].rename({'declaration_date': 'date'}, axis=1)
    dis_data['date'] = dis_data['date'].astype('datetime64[ns]').dt.strftime('%m-%Y')
    # dis_data = dis_data.groupby(['state', 'date']).count()
    dis_data = dis_data.rename({'incident_type': 'disaster_occurrence'}, axis=1)
    dis_data['disaster_occurrence'] = np.ones(dis_data['disaster_occurrence'].shape)
    # dis_data = dis_data.reset_index()

    # print(dis_data)

    dis_data.to_csv('./data/test_disasters_state_month.csv', index=False)

    # filter temperature dataset
    temp_data = pd.read_csv(temp_path)
    temp_data = temp_data[temp_data['Country'] == 'United States'].dropna()  # filter by United States, remove NaNs
    temp_data['date'] = temp_data['dt'].astype('datetime64[ns]').dt.strftime('%m-%Y')  # convert string to date, then convert to year
    temp_data['state'] = temp_data['State'].apply(lambda x: states[x] if x in states else None)  # preprocess state strings
    temp_data = temp_data.dropna()
    temp_data = temp_data.groupby(['date', 'state'])  # group by year then state
    temp_data = temp_data[['AverageTemperature', 'AverageTemperatureUncertainty']].mean().reset_index()  # take average over groups

    # print(temp_data)

    temp_data.to_csv('./data/test_temp_state_month.csv', index=False)

    # join on `Year` and `State`
    df = pd.merge(temp_data, dis_data, on=['date', 'state'], how='left').set_index(['date', 'state'], drop=True)
    df.rename({'AverageTemperature': 'ave_temp', 'AverageTemperatureUncertainty': 'ave_temp_uncertainty'}, axis=1, inplace=True)
    df = df.fillna(0).reset_index()

    print(df)

    df.to_csv('./data/test_disasters_temp_state_month.csv', index=False)


if __name__ == '__main__':
    main()
