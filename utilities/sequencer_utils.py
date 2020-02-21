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


"""
Converts the timestamp to epoch time (in seconds)
Args:
    pd_timestamp:  timestamp to process
Returns:
    epoch time in seconds
"""
def convert_to_epoch(pd_timestamp):
    return pd_timestamp.value//1e9

"""
Converts the epoch time to pandas timestamp
Args:
    epoch_date:  epoch time(in seconds)
Returns:
    Corresponding pandas timestamp
"""
def convert_to_pd_timestamp(epoch_date):
    return pd.Timestamp(dt.datetime.fromtimestamp(epoch_date))

"""
DEBUG function
Args:
    sequence:  sequence of data
    file: file where data is dumped 
Returns:
"""
def print_sample(sequence, file):
    for seq in sequence:
        string = '- DATE: ' + str(convert_to_pd_timestamp(seq['iso-datetime'])) + ', STATION: ' + str(seq['station']) + \
                 ', GHI :' + str(seq['GHI']) + '\n'
        file.write(string)

"""
DEBUG function
Args:
    batch:  batch containing sequences of data
    file: file where data is to be dumped 
Returns:
"""
def print_batch(batch , filename):
    file = open(filename + '.txt','w')
    for sample in batch:
        string = '######## New Sequence ########\n'
        #print(string)
        file.write(string)
        print_sample(sample, file)
    file.close()


"""
Function that computes intersection between 2 sorted lists in O(N)
Args:
    l1: first list
    l2: second list
Returns:
    list containing numbers shared between the two lists in arguments
"""
def intersect_sorted_lists(l1, l2):
    res = []
    id1, id2 = 0, 0
    while id1 < len(l1) and id2 < len(l2):
        op1 = l1[id1]
        op2 = l2[id2]
        if (op1 > op2):
            id2 += 1
        elif (op1 < op2):
            id1 += 1
        else:
            res.append(op1)
            id1 += 1
            id2 += 1
    return res