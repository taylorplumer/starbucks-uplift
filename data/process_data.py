import sys
import pandas as pd
import numpy as np
import math
import json
from pandas.io.json import json_normalize
from datetime import datetime, timedelta
from multiprocessing import Pool
from tqdm import tqdm
tqdm.pandas()



def clean_transcript(df):

    """
    Data preprocessing for transcript dataframe

    Args:
        df: dataframe of transcript file

    Returns:
        df: clean dataframe to be merged with profile dataframe

    """


    df = df.rename(columns={'person': 'id'})
    normalized_value = json_normalize(df['value'])
    normalized_value['offer_id'] = normalized_value['offer_id'].fillna(normalized_value['offer id'])
    normalized_value = normalized_value.drop(columns=['offer id'], axis=1)

    df = df.merge(normalized_value, left_index=True, right_index=True).drop(columns=['value', 'reward'], axis=1)

    return df



def clean_profile(df):

    """
    Data preprocessing for profile dataframe

    Args:
        df: dataframe of profile file

    Returns:
        df: clean dataframe to be merged with transcript dataframe

    """

    def days_from_today(date):

        """
        Function to be used with pandas apply to create "days_as_member" feature

        Args:
            date: date value of 'became_member_on'

        Returns:
           days: integer that is the time delta from today and date value

        """
        delta = datetime.today() - date

        days = delta.days

        return days

    df['became_member_on'] = df['became_member_on'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    df['days_as_member'] = df['became_member_on'].apply(days_from_today)

    return df



def clean_portfolio(df):

    """
    Data preprocessing for portfolio dataframe

    Args:
        df: dataframe of portfolio file

    Returns:
        df: clean dataframe to be merged with the transcript + profile merged dataframe

    """



    df = df.rename(columns={'id': 'offer_id'})

    df.channels = df.channels.fillna('None')

    for element in ['web', 'email', 'mobile', 'social']:
        df[element] = df.channels.apply(lambda x: x.count(element))

    df = df.drop(columns=['channels'], axis=1)

    return df




def create_merged_df(transcript, portfolio, profile):

    """
    Merge three dataframes to create one consolidated dataframe

    Args:
        transcript: dataframe of transcript file that has been cleansed
        portfolio: dataframe of portfolio file that has been cleansed
        profile: dataframe of profile file that has been cleansed

    Returns:
        df: merged dataframe of all three input dataframes that is on an event level unit of analysis

    """
    df = transcript.merge(profile, on='id')

    df = df.merge(portfolio, on='offer_id', how='left')

    return df


def determine_max_time(event, time, duration):

    """
    Calculate maximum time where offer is valid.

    Args:
        event: string value to determine whether calculation is necessary
        time: integer value to be added with duration arguement value
        duration: integer value to be added with time arguement value


    """
    if event == 'offer received':
        max_time = time + duration
        return max_time
    else:
        pass


def set_treatment_outcome(df):

    """

    Create binary values for event of whether treatment and outcome occurred.

    'treatment' binary feature created to indicate wheather treatment occured, aka whether the customer viewed offer.
    'outcome' binary feature created to indicate whether transaction occured during offer duration.


    Args:
        df: merged dataframe resulting from create_merged_df() function

    Returns:
        df: dataframe with 'treatment' and 'outcome' binary features. See 'processed_data_11202019.csv' file for example

    """

    view_df = df.loc[df['event'] == 'offer received']
    def determine_max_time(event, time, duration):
        if event == 'offer received':
            max_time = time + duration
            return max_time
        else:
            pass

    view_df['max_time'] = view_df.progress_apply(lambda x: determine_max_time(x['event'],x['time'], x['duration']), axis=1)
    view_df.loc[view_df['event']=='offer viewed', 'max_time'] = view_df.max_time.fillna(method='ffill')

    view_df.loc[view_df['event']=='offer completed', 'max_time'] = view_df.max_time.fillna(method='ffill')

    def determine_combined(time, max_time, id_):

        timeframe = df.loc[(df['time'] >= time) & (df['time'] <= max_time) & (df['id'] == id_)].reset_index()
        transaction_count = timeframe.event.unique().tolist().count('transaction')
        view_count = timeframe.event.unique().tolist().count('offer viewed')
        outcome = np.where(timeframe.event.unique().tolist().count('transaction')>0, 1, 0)
        treatment = np.where(timeframe.event.unique().tolist().count('offer viewed'),1,0)
        combined = [treatment, outcome]
        return combined


    view_df['combined'] = view_df.progress_apply(lambda x: determine_combined(x['time'], x['max_time'], x['id']), axis=1)

    return view_df

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def save_data(df, filepath):

    """

    Saves file to designated filepath.

    Args:
        df: processed dataframe to load into the database
        filepath: path to store the flat file
    """
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
        #df['max_time'] = df.apply(lambda x: determine_max_time(x['event'],x['time'], x['duration']), axis=1)

        print('Determining treatment and outcome values...')
        processed_df = parallelize_dataframe(df, set_treatment_outcome)
        print(processed_df.dtypes)

        processed_df.to_csv('processed_intermediate_01272020.csv')
        offer_received = df.loc[df['event'] == 'offer received']
        combined_df = pd.DataFrame(processed_df.combined.astype(str).str.strip('[]').str.split(',', expand=True).values, columns=[ 'treatment','outcome'])
        output_df = offer_received.merge(combined_df, on=offer_received.index).drop(columns=['key_0'], axis=1)

        print('Saving data...')
        save_data(output_df, filepath)

        print('Cleaned data saved as csv!')

    else:
        print('Two arguements required. Provide filepath as second arguement')


if __name__ == '__main__':
    main()
