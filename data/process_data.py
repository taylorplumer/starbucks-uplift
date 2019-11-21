import sys
import pandas as pd
import numpy as np
import math
import json
from pandas.io.json import json_normalize
from datetime import datetime, timedelta
from tqdm import tqdm
tqdm.pandas()


def clean_transcript(df):
    
    df = df.rename(columns={'person': 'id'})
    normalized_value = json_normalize(df['value'])
    normalized_value['offer_id'] = normalized_value['offer_id'].fillna(normalized_value['offer id'])
    normalized_value = normalized_value.drop(columns=['offer id'], axis=1)
    
    df = df.merge(normalized_value, left_index=True, right_index=True).drop(columns=['value', 'reward'], axis=1)
    
    return df   



def clean_profile(df):
    
    def days_from_today(date):
        delta = datetime.today() - date

        days = delta.days

        return days
    
    df['became_member_on'] = df['became_member_on'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    df['days_as_member'] = df['became_member_on'].apply(days_from_today)
    
    return df



def clean_portfolio(df):
    
    df = df.rename(columns={'id': 'offer_id'})
        
    df.channels = df.channels.fillna('None')
    
    for element in ['web', 'email', 'mobile', 'social']:
        df[element] = df.channels.apply(lambda x: x.count(element))
    
    df = df.drop(columns=['channels'], axis=1)
    
    return df




def create_merged_df(transcript, portfolio, profile):
    df = transcript.merge(profile, on='id')
    
    df = df.merge(portfolio, on='offer_id', how='left')
    
    return df
    

def determine_max_time(event, time, duration):
    if event == 'offer received':
        max_time = time + duration
        return max_time
    else:
        pass


def set_treatment_outcome(df):
    
    id_list = df.id.unique().tolist()

    df['outcome'] = [np.nan for i in range(df.shape[0])]
    df['treatment'] = [np.nan for i in range(df.shape[0])]

    for id_ in tqdm(id_list):
        id_df = df.loc[df['id'].isin([id_])]
        id_df = id_df[['event', 'offer_type', 'id', 'time', 'duration', "offer_id"]].sort_values(by=['id', 'time'])

        id_viewed_df = id_df.loc[id_df['event'] == 'offer received'].reset_index()


        for i in range(id_viewed_df.shape[0]):
            offer_id = id_viewed_df.offer_id[i]
            offer_type = id_viewed_df.offer_type[i]
            time = id_viewed_df.time[i]
            duration = id_viewed_df.duration[i]
            max_time = time + duration
            timeframe = id_df.loc[(id_df['time'] >= time) & (id_df['time'] <= max_time)].reset_index()
            transaction_count = timeframe.event.unique().tolist().count('transaction')
            view_count = timeframe.event.unique().tolist().count('offer viewed')
            row_position = df.loc[(df['offer_id'] == offer_id) & (df['time'] == time) & (df['event'] == 'offer received') & (df['id'] == id_)].index[0]

            if transaction_count > 0:
                df.iloc[row_position, -2] = 1 # ensure that outcome is the second to last column
            else:
                df.iloc[row_position, -2] = 0
            if view_count > 0:
                df.iloc[row_position, -1] = 1 # ensure that treatment is the last column
            else:
                df.iloc[row_position, -1] = 0
    
    df = df.loc[df['event'] == 'offer received']
    return df

def save_data(df, filepath):
    df.to_csv(filepath, index=False)


def main():
    
    if len(sys.argv) == 2:
        
        filepath = sys.argv[1]
          
        print('Loading data..')
        portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
        profile = pd.read_json('data/profile.json', orient='records', lines=True)
        transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

        print('Cleaning data..')
        transcript = clean_transcript(transcript)
        profile = clean_profile(profile)
        portfolio = clean_portfolio(portfolio)

        df = create_merged_df(transcript, portfolio, profile)
        df['max_time'] = df.apply(lambda x: determine_max_time(x['event'],x['time'], x['duration']), axis=1)

        print('Determining treatment and outcome values...')
        output_df = set_treatment_outcome(df)

        print('Saving data...')
        save_data(output_df, filepath)

        print('Cleaned data saved as csv!')
        
    else:
        print('Two arguements required. Provide filepath as second arguement')
    
    
if __name__ == '__main__':
    main()

