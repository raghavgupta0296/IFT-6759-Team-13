import pandas as pd

MIN_TIME = 15


"""
Returns the same time at next day given a timestamp
Args:
    start_time:  timestamp 
Returns:
    Timestamp corresponding to next day(same time)
"""
def next_day(start_time):
    return start_time + pd.Timedelta(str(1) + ' days')


"""
Computes the next date given an offset 
Args:
    start_time:  timestamp where we start
    offset: delay to add to the current time
Returns:
    (current + 15 min * offset)
"""
def next_time(start_time, offset):
    return start_time + pd.Timedelta(str(offset * MIN_TIME) + ' minutes')


"""
Converts the local time(h:m:s) to seconds
Args:
    pd_timestamp:  timestamp to process
Returns:
    local time in seconds
"""
def time_in_seconds(pd_timestamp):
    df_time = pd.to_datetime(pd_timestamp)
    return (df_time.hour*60+df_time.minute)*60 + df_time.second




