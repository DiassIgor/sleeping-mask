import requests
import datetime
import collections

import pandas as pd
import numpy as np
from tqdm import tqdm

def data_request(start, end):   
    """
    Description
    -----------
    This function gets requests from the sleep and heart routes, in a specific date range

    Parameters
    ----------
    start : <string>
        Beginning of the interval. The day is in the format YYYY-MM-DD

    end : <string>
        End of the interval. The day is in the format YYYY-MM-DD

    Returns
    -------
    sleep_response : <class 'requests.models.Response'>
        Response from the sleep route

    heart_response : <class 'requests.models.Response'>
        Response from the heart route
    """
    
    # Personal Access Token used to access the user data via the Oura Cloud API
    headers = {'Authorization': 'Bearer CZQY7GXLYM4JRPEWO4SYGLCRFG5BXD5K'}
    
    # Parent route
    url = f'https://api.ouraring.com/v2/usercollection/'
    
    # Define the parameters of each request
    sleep_params = {'start_date': start, 
                    'end_date': end}    
    heart_params={ 'start_datetime': f'{start}T00:00:01+03:00', 
                   'end_datetime': f'{end}T23:59:59+03:00'}
    
    # Request Sleep data and Heart data
    sleep_response = requests.request('GET', url+'sleep', headers = headers, params = sleep_params)
    heart_response = requests.request('GET', url+'heartrate', headers = headers, params = heart_params)
    
    return sleep_response, heart_response


def heart_route_preprocessing(response):
    """
    Description
    -----------
    This function extracts the bpm and time from the json 

    Parameters
    ----------
    response : <class 'requests.models.Response'>
        Response from the heart route

    Returns
    -------
    heart_data : <class 'pandas.core.frame.DataFrame'>
        DataFrame with integer index and ['time', 'bpm'] columns 
    """
    
    # List that will be filled with the response data
    time, bpm = [], []

    # Extract the information from the response
    for data in response.json()['data']:
        
        bpm.append(data['bpm'])
        time.append(data['timestamp'])

    # Create a Empty DataFrame
    heart_data = pd.DataFrame(columns = ["time", "bpm"])
    
    # Store the response data in the DataFrame
    heart_data["time"], heart_data["bpm"] = time, bpm

    return heart_data


def time_preprocessing(time):
    """
    Description
    -----------
    This function changes the format of a date and adjusts the time according to the Latvian or Brazilian timezone. OBS: These timezones were chosen according to the countries in which Igor lived.

    Parameters
    ----------
    time : <str>
        Time in the format YYYY-MM-DDThh:mm:ss+00:00 (Example: 2023-07-24T21:04:37+00:00)

    Returns
    -------
    new_time : <class 'datetime.datetime'>
        Time in the format YYYY-MM-DD hh:mm:ss±03:00 (Example: 2023-07-25 00:04:37+03:00)
    """

    # Set the Latvian and Brazilian timezones (UTC +3 and UTC -3, respectively) 
    LV_TIMEZONE = datetime.timezone(offset = datetime.timedelta(hours=3))
    BR_TIMEZONE = datetime.timezone(offset = datetime.timedelta(hours=-3))

    # Covert string to Datetime
    new_time = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S%z")
    
    # Choose the correct timezone, according with the date that Igor comeback to Brazil
    if (new_time <= datetime.datetime(2023, 8, 24, tzinfo=datetime.timezone.utc)):
        new_time = new_time.astimezone(LV_TIMEZONE)
        
    else:
        new_time = new_time.astimezone(BR_TIMEZONE)
    
    return new_time


def groups_5min(data):
    """
    Description
    -----------
    Every 5 minutes, the Oura Ring measures the heart rate for 60 consecutive seconds. However, only the reliable measures are stored in the API. 
    Therefore, this function groups all the heart rates that were collected in the same 60s batch.

    Parameters
    ----------
    data : <class 'pandas.core.frame.DataFrame'>
        Heart data. DataFrame with integer index and ['time', 'bpm'] columns

    Returns
    -------
    new_data : <class 'pandas.core.frame.DataFrame'>
        DataFrame with integer index and ['time', 'bpm','state'] columns
    """
    
    # Set the time difference between the current row and the previous one
    mask = heart_data['time'].diff().dt.seconds

    # Every time that a time diff is greater than 60s, add +1 to the label 
    mask = mask.gt(60).cumsum()

    # Group the batches according to the mask, maintaining the time of the first measure, and the mean of the bpm    
    new_data = data.groupby(mask, as_index=True)[['time','bpm']].agg({'time':'first', 'bpm':'mean'}).round(1)
    
    # Set the awake state (will be important during the 4-stages classification)
    new_data['state'] = 'awake' 
    
    return new_data


def heart_rate_extractor(day_data):
    """
    Description
    -----------
    This function extracts the heart rate from the sleep route and gives back the DataFrame in the same format as the heart route Dataframe. 
   
    Parameters
    ----------
    day_data : <dict>
        JSON with data of one-night sleep time

    Returns
    -------
    new_data : <class 'pandas.core.frame.DataFrame'>
        Heart data during the sleep time. DataFrame with integer index and ['time', 'bpm','state'] columns
    """
    
    # Create the columns 
    time = []
    bpm = day_data['heart_rate']['items']
    state = ['sleep']*len(bpm)
    
    # Extract the start and end of the sleep time
    start = datetime.datetime.strptime(day_data['bedtime_start'], "%Y-%m-%dT%H:%M:%S%z")
    end = datetime.datetime.strptime(day_data['bedtime_end'], "%Y-%m-%dT%H:%M:%S%z")
    
    # Create a spaced timelist within the sleep time interval
    aux = start
    while aux < end:
        time.append(aux)
        aux += datetime.timedelta(minutes=5)
        
    # Padding of the time length according to bpm length
    while len(bpm) != len(time):
        time.pop()
        
    # Create a DataFrame with the sleep data of a unique day
    heart_data = pd.DataFrame(columns = ["time", "bpm", "state"])
    heart_data["time"], heart_data["bpm"], heart_data["state"] = time, bpm, state

    return heart_data


def sleep_route_preprocessing(response):
    """
    Description
    -----------
    This function iterate over the days to extracts the sleep data with the heart_rate_extractor

    Parameters
    ----------
    response : <class 'requests.models.Response'>
        Response from the sleep route

    Returns
    -------
    sleep_data : <class 'pandas.core.frame.DataFrame'>
        DataFrame with integer index and ['time', 'bpm', 'state'] columns 
        
    start_bedtime : <list>
        List of with the start bedtime of each day 
    
    end_bedtime : <list>
        List of with the end bedtime of each day
    """
    
    df_list = []
    start_bedtime = []
    end_bedtime = []

    for day_data in response.json()['data']:

        if day_data['heart_rate'] != None:
            if len(day_data['heart_rate']['items']) > 40:

                df_list.append(heart_rate_extractor(day_data))
                start_bedtime.append(day_data['bedtime_start'])
                end_bedtime.append(day_data['bedtime_end'])
    
    sleep_data = pd.concat(df_list)
    
    return sleep_data, start_bedtime, end_bedtime


def time_rounder(full_data):
    """
    Description
    -----------
    This function round the time in multiple hours of 5 minutes (Ex: 00h00, 00h05, 00h10, 00h15, and so on).

    Parameters
    ----------
    full_data : <class 'pandas.core.frame.DataFrame'>
        Sleep and Heart route concatenated. DataFrame with integer index and ['time', 'bpm', 'state'] columns

    Returns
    -------
    full_data : <class 'pandas.core.frame.DataFrame'>
        Full data with the time rounded to 5 minutes interval. DataFrame with integer index and ['time', 'bpm', 'state'] columns
    """
    
    # Round the time in multiple hours of 5 minutes
    full_data['new_time'] = full_data['time'].round('5min')
    
    # Take all duplicate time after the round
    duplicate = [item for item, count in collections.Counter(full_data["new_time"]).items() if count > 1]
    
    for dupl in duplicate:
        
        idx = full_data[full_data['new_time'] == dupl].index
        
        # Round down the first duplicated row
        floor_round = full_data.iloc[idx[0]]['time'].floor('5min')
        
        # Round up the second duplicated row
        ceil_round = full_data.iloc[idx[1]]['time'].ceil('5min')

        # Replace the first duplicated if the new round doesn't conflict with the previous value
        if floor_round != full_data.iloc[idx[0]-1]['new_time']:
            full_data.at[idx[0],'new_time'] = floor_round

        # Replace the second duplicated if the new round doesn't conflict with the next value
        elif ceil_round != full_data.iloc[idx[1]+1]['new_time']:
            full_data.at[idx[1],'new_time'] = ceil_round

        # Exclude the first duplicated if the two replace methods didn't work
        else:
            full_data = full_data.drop(axis=0,index=idx[0]).reset_index(drop=True)

    # Maintain the same format as the DataFrame from the input
    full_data = full_data.drop('time',axis=1)
    full_data = full_data[['new_time','bpm','state']].rename(columns={"new_time": "time"})
    
    return full_data


def gap_filler(full_data, start, end):
    """
    Description
    -----------
    This function round the time in multiple hours of 5 minutes (Ex: 00h00, 00h05, 00h10, 00h15, and so on).

    Parameters
    ----------
    full_data : <class 'pandas.core.frame.DataFrame'>
        Sleep and Heart route concatenated. DataFrame with integer index and ['time', 'bpm', 'state'] columns

    start : <string>
        Beginning of the interval. The day is in the format YYYY-MM-DD

    end : <string>
        End of the interval. The day is in the format YYYY-MM-DD

    Returns
    -------
    full_data : <class 'pandas.core.frame.DataFrame'>
        Full data with iserted gaps filled with NaN values. DataFrame with datetime64 index and ['bpm', 'state'] columns
    """

    #Add the hour in the start and end time
    start_time = datetime.datetime.strptime(start + "T00:00:00+03:00", "%Y-%m-%dT%H:%M:%S%z")
    end_time = datetime.datetime.strptime(end + "T23:55:00+03:00", "%Y-%m-%dT%H:%M:%S%z")
    
    # Create a spaced timelist within the sleep time interval
    time_list = set()
    aux = start_time
    
    while aux <= end_time:
        time_list.add(aux)
        aux += datetime.timedelta(minutes=5)
    
    # Take just the times that are not in the data 
    gaps = time_list - time_list.intersection(full_data['time'])
    
    # Concatenate the data with the new rolls
    new_rows = {'time': list(gaps), 'bpm':[None]*len(gaps), 'state':[None]*len(gaps)}
    full_data = pd.concat([full_data, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Set the index as a timestamp
    full_data.index = full_data['time']
    full_data = full_data.drop('time', axis=1).sort_index()
    
    # Fill the None state with the state of the previous roll
    #full_data['state'] = full_data['state'].ffill()
    
    return full_data


def get_time_label(bedtime):
    """
    Description
    -----------
    This function cast the start or end bedtime list into a DataFrame with a rounded time

    Parameters
    ----------
    bedtime : list<str>
        List with all start bedtime or end bedtime. The elements are in the format YYYY-MM-DDThh:mm:ss+hh:mm 

    Returns
    -------
    df_label : <class 'pandas.core.frame.DataFrame'>
        DataFrame with integer index and ['time'] column with all bedtime values
    """
    
    # String to datetime with an offset to adjust the labels 
    label = [datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S%z") + datetime.timedelta(minutes=5) for time in bedtime] 
    
    # Create a DataFrame with the bedtime list
    df_label = pd.DataFrame(label, columns = ["time"])
    
    # Round the time to fit with our standardization
    df_label['time'] = df_label['time'].round('5min')

    return df_label


def data_labelling(data, start_bedtime, end_bedtime):
    """
    Description
    -----------
    This create a column with a binary classification where:
        0 = active time
        1 = sleep time

    Parameters
    ----------
    data : <class 'pandas.core.frame.DataFrame'>
       Full data with 5 minutes intervals. DataFrame with datetime64 index and ['bpm', 'state'] columns

    start_bedtime : list <str>
        List with all times that the user went to bed. The times are in the format YYYY-MM-DDThh:mm:ss+hh:mm 

    end_bedtime : list <str>
        List with all times that the user get up. The times are in the format YYYY-MM-DDThh:mm:ss+hh:mm 

    Returns
    -------
    data : <class 'pandas.core.frame.DataFrame'>
        Full data with with a binary label column. DataFrame with datetime64 index and ['bpm', 'state', 'label'] columns
    """
    
    # Cast the bedtime lists into a DataFrame with a rounded time
    df_get_up = get_time_label(end_bedtime)
    df_lay_down = get_time_label(start_bedtime)
    
    # Mark all rows where the user got up or lay down
    data.loc[data.index.isin(df_get_up['time']), 'label'] = 0
    data.loc[data.index.isin(df_lay_down['time']), 'label'] = 1
    
    # Fill the None label with the state of the previous roll
    data['label'] = data['label'].ffill()
    
    # Remove the rows which the ffill propagation didn't work
    data = data.dropna(subset=['label'])
    
    return data


def state_filler(data):    
    """
    Description
    -----------
    This function fills the None state according with the label: 'awake' for active time and 'sleep' for sleep time

    Parameters
    ----------
    data : <class 'pandas.core.frame.DataFrame'>
        Data with None values in the state column. DataFrame with datetime64 index and ['bpm', 'state', 'label'] columns

    Returns
    -------
   data : <class 'pandas.core.frame.DataFrame'>
        Data without None values in the state column. DataFrame with datetime64 index and ['bpm', 'state', 'label'] columns
    """
    
    
    for idx, row in data.iterrows():
        if row['state'] == None:
            
            # Become 'awake' in the active time
            if row['label'] == 0:
                data.at[idx, 'state'] = 'awake'
            
            # Become 'sleep' in the sleep time
            elif row['label'] == 1:
                data.at[idx, 'state'] = 'sleep'   
    
    return data


def day_batcher(df):
    """
    Description
    -----------
    This function take the data and split into daily batches

    Parameters
    ----------
    df : <class 'pandas.core.frame.DataFrame'>
        Complete data. DataFrame with datetime64 index and ['bpm', 'state', 'label'] columns

    Returns
    -------
   data : list <class 'pandas.core.frame.DataFrame'>
        List with the complete data split into daily batches
    """
        
    # Create an auxiliar DataFrame with integer index 
    aux = df.copy().reset_index()
    
    # Take the index where the label change from 1 to 0
    day_end = aux[aux['label'].diff() == -1].index 
    
    # Split into daily batches
    day_batch = []   
    for idx in range(len(day_end) - 1):
        if idx == 0:
            day_batch.append(df.iloc[0:day_end[idx]])
        else:
            day_batch.append(df.iloc[day_end[idx]+1: day_end[idx+1]])
            
    return day_batch