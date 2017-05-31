import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse

dataset = pd.read_csv(u'/home/haven/Tianchi_power/Tianchi_power.csv')

def test(group):
    group.drop('user_id', axis=1, inplace=True)
    group.index = pd.to_datetime(group.record_date)
    group.drop('record_date',axis=1, inplace=True)
    group = group.resample('D')
    group.fillna(value=0, axis=0,inplace=True)
    return group
handledataset = dataset.groupby('user_id').apply(test)
handledataset.reset_index(inplace=True)
def getDayOfWeek(x):
#     date_parse = parse(x)
    return x.weekday()+1
handledataset['day_of_week'] = handledataset.record_date.apply(getDayOfWeek)
handledataset.to_csv(u'/home/haven/Tianchi_power/Tianchi_power_resample_addDayOfWeek.csv', header=True,index=False)
