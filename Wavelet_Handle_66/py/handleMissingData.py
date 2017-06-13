import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse

def getDayOfWeek(x):
#     date_parse = parse(x)
    return x.weekday()+1

def resample1(group):
    group.drop('user_id', axis=1, inplace=True)
    group.index = pd.to_datetime(group.record_date)
    group.drop('record_date',axis=1, inplace=True)
    group = group.resample('D')
    group.fillna(value=0, axis=0,inplace=True)
    return group

def resample2(group):
    group.drop('user_id', axis=1, inplace=True)
    group.index = pd.to_datetime(group.record_date)
    group.drop('record_date',axis=1, inplace=True)
    return group
print "loading originDataset ..........."
dataset1 = pd.read_csv(u'/home/haven/Tianchi_power/Tianchi_power.csv')
dataset2 = pd.read_csv(u'/home/haven/Tianchi_power/Tianchi_power_9month.csv')

print "handle dataset1 ..........."
handledataset1 = dataset1.groupby('user_id').apply(resample1)
handledataset1.reset_index(inplace=True)
handledataset1['day_of_week'] = handledataset1.record_date.apply(getDayOfWeek)
handledataset1.to_csv(u'/home/haven/Tianchi_power/Tianchi_power_Hanle_addDayOfWeek.csv', header=True,index=False)

print "handle dataset2 ..........."
handledataset2 = datase2.groupby('user_id').apply(resample2)
handledataset2.reset_index(inplace=True)
handledataset2['day_of_week'] = handledataset2.record_date.apply(getDayOfWeek)
handledataset2.to_csv(u'/home/haven/Tianchi_power/Tianchi_power_9month_addofweek.csv', header=True,index=False)

print "make finalDataset ..........."
mergedataset = pd.concat([olddataset,newdataset], ignore_index=True, axis=0)
mergedataset.to_csv(u'/home/haven/Tianchi_power/Tianchi_power_newold_mergeDataset.csv', header=True, index=False)

