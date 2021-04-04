import re

import pandas as pd


def preproc_state(s, states=None, single_state=False):
    # converts input string `s` of some string containing one or multiple states into a list of unique states
    if states is None:
        states = []

    s = s.replace('(', ',').replace(')', ',').split(',')  # assume csv, replace parentheses with comma, split on comma
    s = [re.sub(r'^\W+|\W+$', '', w.lower()) for w in s]  # convert strings to lowercase, remove leading and trailing nonalphanumeric characters
    states_s = set()  # unique set
    for v in s:
        state = ' '.join([w.capitalize() for w in v.split()])  # capitalize candidate state string

        if state in states:
            states_s.add(state)

    if states_s:
        if single_state:
            return list(states_s)[0]
        else:
            return list(states_s)
    else:
        return None


def main():
    dis_path = './data/emdat_public_2021_04_01_query_uid-s3dTaw.xlsx'
    temp_path = './data/GlobalLandTemperaturesByState.csv'
    states_path = './data/states.txt'

    # load list of states
    states = []
    with open(states_path) as f:
        states = f.read().splitlines()

    # filter disaster dataset
    dis_data = pd.read_excel(dis_path)
    dis_data = dis_data[dis_data['ISO'] == 'USA'].dropna(subset=['Location'])  # show only USA with nonempty `Location`
    dis_data['Location'] = dis_data['Location'].apply(preproc_state, args=[states])  # preprocess strings in `Location` (see `preproc_state` for more details)
    dis_data.dropna(subset=['Location'], inplace=True)  # remove resulting empty `Location` values
    dis_data.rename(columns={'Location': 'State'}, inplace=True)  # rename `Location` to `State
    dis_data = dis_data.explode('State')  # expand unique `State` values into separate rows
    dis_data.reset_index(drop=True, inplace=True)  # reset row index

    # dis_data.to_csv('./data/scrubbed1.csv', index=False)

    # filter temperature dataset
    temp_data = pd.read_csv(temp_path)
    temp_data = temp_data[temp_data['Country'] == 'United States'].dropna()  # filter by United States, remove NaNs
    temp_data['Year'] = temp_data['dt'].astype('datetime64[ns]').dt.year  # convert string to date, then convert to year
    temp_data['State'] = temp_data['State'].apply(preproc_state, args=[states, True])  # preprocess state strings (see `preproc_state` for more details)
    temp_data = temp_data.groupby(['Year', 'State'])  # group by year then state
    temp_data = temp_data['AverageTemperature', 'AverageTemperatureUncertainty'].mean()  # take average over groups

    # temp_data.to_csv('./data/scrubbed2.csv')

    # join on `Year` and `State`
    df = pd.merge(temp_data, dis_data, on=['Year', 'State'], how='inner').set_index(['Year', 'State'], drop=True)

    # print(df)

    df.to_csv('./data/yearly_temp_disaster_by_state.csv')


if __name__ == '__main__':
    main()
