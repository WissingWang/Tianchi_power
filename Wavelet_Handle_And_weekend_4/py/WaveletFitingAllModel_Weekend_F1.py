import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.externals import joblib
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn import metrics
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
import AllModelWaveletTransformByDOW

print "loading dataset .............." 
handledataset = pd.read_csv(u'/home/haven/Tianchi_power/Tianchi_power__boxhandle15_DayOfWeek.csv')

print "transform date to datetime .............."
handledataset.record_date = pd.to_datetime(handledataset.record_date)

print "select train1 dataset ............."
train1 = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*66, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*78, unit='D')))]

train1_noWeekend = train1[(train1.day_of_week>=1) & (train1.day_of_week<=5)]
train1_Weekend = train1[(train1.day_of_week==6) | (train1.day_of_week==7)]

print "============ train1_noWeekend ==========="
train1_noWeekend_MeanStdSum = train1_noWeekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train1_noWeekend_MeanStdSum['power_rate'] = train1_noWeekend_MeanStdSum.power_sum / train1_noWeekend_MeanStdSum.power_sum.sum()
train1_noWeekend_DOW_MeanStdSum = train1_noWeekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train1_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train1_noWeekend_DOW_MeanStdSumAllsum = train1_noWeekend_DOW_MeanStdSum.merge(train1_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train1_noWeekend_DOW_MeanStdSumAllsum['DOW_power_rate'] = train1_noWeekend_DOW_MeanStdSumAllsum.DOW_power_sum / train1_noWeekend_DOW_MeanStdSumAllsum.DOW_allsum
train1_noWeekend_mergeDataset = pd.merge(train1_noWeekend_DOW_MeanStdSumAllsum, train1_noWeekend_MeanStdSum, on='user_id', how='left')

train1_noWeekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train1_noWeekend)
train1_noWeekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train1_noWeekend_mergeDataset_add = pd.merge(train1_noWeekend_mergeDataset,train1_noWeekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train1_mergeDataset_add = train1_mergeDataset
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train1_noWeekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)


print "============ train1_Weekend ==========="
train1_Weekend_MeanStdSum = train1_Weekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train1_Weekend_MeanStdSum['power_rate'] = train1_Weekend_MeanStdSum.power_sum / train1_Weekend_MeanStdSum.power_sum.sum()
train1_Weekend_DOW_MeanStdSum = train1_Weekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train1_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train1_Weekend_DOW_MeanStdSumAllsum = train1_Weekend_DOW_MeanStdSum.merge(train1_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train1_Weekend_DOW_MeanStdSumAllsum['DOW_power_rate'] = train1_Weekend_DOW_MeanStdSumAllsum.DOW_power_sum / train1_Weekend_DOW_MeanStdSumAllsum.DOW_allsum
train1_Weekend_mergeDataset = pd.merge(train1_Weekend_DOW_MeanStdSumAllsum, train1_Weekend_MeanStdSum, on='user_id', how='left')

train1_Weekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train1_Weekend)
train1_Weekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train1_Weekend_mergeDataset_add = pd.merge(train1_Weekend_mergeDataset,train1_Weekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train1_mergeDataset_add = train1_mergeDataset
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train1_Weekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)

print "make train1_Y .........."
train1_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*78, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*79, unit='D')))]
train1_noWeekend_Y = train1_Y[(train1_Y.day_of_week>=1) & (train1_Y.day_of_week<=5)]
train1_Weekend_Y = train1_Y[(train1_Y.day_of_week==6) | (train1_Y.day_of_week==7)]

final_train1_noWeekend = pd.merge(train1_noWeekend_mergeDataset_add,train1_noWeekend_Y,on=['user_id', 'day_of_week'], how='left')
final_train1_Weekend = pd.merge(train1_Weekend_mergeDataset_add,train1_Weekend_Y,on=['user_id', 'day_of_week'], how='left')



print "select train2 dataset ............."
train2 = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*67, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*79, unit='D')))]

train2_noWeekend = train2[(train2.day_of_week>=1) & (train2.day_of_week<=5)]
train2_Weekend = train2[(train2.day_of_week==6) | (train2.day_of_week==7)]

print "============ train2_noWeekend ==========="
train2_noWeekend_MeanStdSum = train2_noWeekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train2_noWeekend_MeanStdSum['power_rate'] = train2_noWeekend_MeanStdSum.power_sum / train2_noWeekend_MeanStdSum.power_sum.sum()
train2_noWeekend_DOW_MeanStdSum = train2_noWeekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train2_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train2_noWeekend_DOW_MeanStdSumAllsum = train2_noWeekend_DOW_MeanStdSum.merge(train2_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train2_noWeekend_DOW_MeanStdSumAllsum['DOW_power_rate'] = train2_noWeekend_DOW_MeanStdSumAllsum.DOW_power_sum / train2_noWeekend_DOW_MeanStdSumAllsum.DOW_allsum
train2_noWeekend_mergeDataset = pd.merge(train2_noWeekend_DOW_MeanStdSumAllsum, train2_noWeekend_MeanStdSum, on='user_id', how='left')

train2_noWeekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train2_noWeekend)
train2_noWeekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train2_noWeekend_mergeDataset_add = pd.merge(train2_noWeekend_mergeDataset,train2_noWeekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train1_mergeDataset_add = train1_mergeDataset
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train2_noWeekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)


print "============ train2_Weekend ==========="
train2_Weekend_MeanStdSum = train2_Weekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train2_Weekend_MeanStdSum['power_rate'] = train2_Weekend_MeanStdSum.power_sum / train2_Weekend_MeanStdSum.power_sum.sum()
train2_Weekend_DOW_MeanStdSum = train2_Weekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train2_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train2_Weekend_DOW_MeanStdSumAllsum = train2_Weekend_DOW_MeanStdSum.merge(train2_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train2_Weekend_DOW_MeanStdSumAllsum['DOW_power_rate'] = train2_Weekend_DOW_MeanStdSumAllsum.DOW_power_sum / train2_Weekend_DOW_MeanStdSumAllsum.DOW_allsum
train2_Weekend_mergeDataset = pd.merge(train2_Weekend_DOW_MeanStdSumAllsum, train2_Weekend_MeanStdSum, on='user_id', how='left')

train2_Weekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train2_Weekend)
train2_Weekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train2_Weekend_mergeDataset_add = pd.merge(train2_Weekend_mergeDataset,train2_Weekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train1_mergeDataset_add = train1_mergeDataset
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train2_Weekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)

print "make train2_Y .........."
train2_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*79, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*80, unit='D')))]
train2_noWeekend_Y = train2_Y[(train2_Y.day_of_week>=1) & (train2_Y.day_of_week<=5)]
train2_Weekend_Y = train2_Y[(train2_Y.day_of_week==6) | (train2_Y.day_of_week==7)]

final_train2_noWeekend = pd.merge(train2_noWeekend_mergeDataset_add,train2_noWeekend_Y,on=['user_id', 'day_of_week'], how='left')
final_train2_Weekend = pd.merge(train2_Weekend_mergeDataset_add,train2_Weekend_Y,on=['user_id', 'day_of_week'], how='left')



print "select train3 dataset ............."
train3 = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*68, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*80, unit='D')))]

train3_noWeekend = train3[(train3.day_of_week>=1) & (train3.day_of_week<=5)]
train3_Weekend = train3[(train3.day_of_week==6) | (train3.day_of_week==7)]

print "============ train3_noWeekend ==========="
train3_noWeekend_MeanStdSum = train3_noWeekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train3_noWeekend_MeanStdSum['power_rate'] = train3_noWeekend_MeanStdSum.power_sum / train3_noWeekend_MeanStdSum.power_sum.sum()
train3_noWeekend_DOW_MeanStdSum = train3_noWeekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train3_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train3_noWeekend_DOW_MeanStdSumAllsum = train3_noWeekend_DOW_MeanStdSum.merge(train3_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train3_noWeekend_DOW_MeanStdSumAllsum['DOW_power_rate'] = train3_noWeekend_DOW_MeanStdSumAllsum.DOW_power_sum / train3_noWeekend_DOW_MeanStdSumAllsum.DOW_allsum
train3_noWeekend_mergeDataset = pd.merge(train3_noWeekend_DOW_MeanStdSumAllsum, train3_noWeekend_MeanStdSum, on='user_id', how='left')

train3_noWeekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train3_noWeekend)
train3_noWeekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train3_noWeekend_mergeDataset_add = pd.merge(train3_noWeekend_mergeDataset,train3_noWeekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train1_mergeDataset_add = train1_mergeDataset
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train3_noWeekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)


print "============ train3_Weekend ==========="
train3_Weekend_MeanStdSum = train3_Weekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train3_Weekend_MeanStdSum['power_rate'] = train3_Weekend_MeanStdSum.power_sum / train3_Weekend_MeanStdSum.power_sum.sum()
train3_Weekend_DOW_MeanStdSum = train3_Weekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train3_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train3_Weekend_DOW_MeanStdSumAllsum = train3_Weekend_DOW_MeanStdSum.merge(train3_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train3_Weekend_DOW_MeanStdSumAllsum['DOW_power_rate'] = train3_Weekend_DOW_MeanStdSumAllsum.DOW_power_sum / train3_Weekend_DOW_MeanStdSumAllsum.DOW_allsum
train3_Weekend_mergeDataset = pd.merge(train3_Weekend_DOW_MeanStdSumAllsum, train3_Weekend_MeanStdSum, on='user_id', how='left')

train3_Weekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train3_Weekend)
train3_Weekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train3_Weekend_mergeDataset_add = pd.merge(train3_Weekend_mergeDataset,train3_Weekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train1_mergeDataset_add = train1_mergeDataset
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train3_Weekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)

print "make train3_Y .........."
train3_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*80, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*81, unit='D')))]
train3_noWeekend_Y = train3_Y[(train3_Y.day_of_week>=1) & (train3_Y.day_of_week<=5)]
train3_Weekend_Y = train3_Y[(train3_Y.day_of_week==6) | (train3_Y.day_of_week==7)]

final_train3_noWeekend = pd.merge(train3_noWeekend_mergeDataset_add,train3_noWeekend_Y,on=['user_id', 'day_of_week'], how='left')
final_train3_Weekend = pd.merge(train3_Weekend_mergeDataset_add,train3_Weekend_Y,on=['user_id', 'day_of_week'], how='left')




print "select train4 dataset ............."
train4 = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*69, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*81, unit='D')))]
train4_noWeekend = train4[(train4.day_of_week>=1) & (train4.day_of_week<=5)]
train4_Weekend = train4[(train4.day_of_week==6) | (train4.day_of_week==7)]

print "============ train4_noWeekend ==========="
train4_noWeekend_MeanStdSum = train4_noWeekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train4_noWeekend_MeanStdSum['power_rate'] = train4_noWeekend_MeanStdSum.power_sum / train4_noWeekend_MeanStdSum.power_sum.sum()
train4_noWeekend_DOW_MeanStdSum = train4_noWeekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train4_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train4_noWeekend_DOW_MeanStdSumAllsum = train4_noWeekend_DOW_MeanStdSum.merge(train4_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train4_noWeekend_DOW_MeanStdSumAllsum['DOW_power_rate'] = train4_noWeekend_DOW_MeanStdSumAllsum.DOW_power_sum / train4_noWeekend_DOW_MeanStdSumAllsum.DOW_allsum
train4_noWeekend_mergeDataset = pd.merge(train4_noWeekend_DOW_MeanStdSumAllsum, train4_noWeekend_MeanStdSum, on='user_id', how='left')

train4_noWeekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train4_noWeekend)
train4_noWeekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train4_noWeekend_mergeDataset_add = pd.merge(train4_noWeekend_mergeDataset,train4_noWeekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train1_mergeDataset_add = train1_mergeDataset
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train4_noWeekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)


print "============ train4_Weekend ==========="
train4_Weekend_MeanStdSum = train4_Weekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train4_Weekend_MeanStdSum['power_rate'] = train4_Weekend_MeanStdSum.power_sum / train4_Weekend_MeanStdSum.power_sum.sum()
train4_Weekend_DOW_MeanStdSum = train4_Weekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train4_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train4_Weekend_DOW_MeanStdSumAllsum = train4_Weekend_DOW_MeanStdSum.merge(train4_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train4_Weekend_DOW_MeanStdSumAllsum['DOW_power_rate'] = train4_Weekend_DOW_MeanStdSumAllsum.DOW_power_sum / train4_Weekend_DOW_MeanStdSumAllsum.DOW_allsum
train4_Weekend_mergeDataset = pd.merge(train4_Weekend_DOW_MeanStdSumAllsum, train4_Weekend_MeanStdSum, on='user_id', how='left')

train4_Weekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train4_Weekend)
train4_Weekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train4_Weekend_mergeDataset_add = pd.merge(train4_Weekend_mergeDataset,train4_Weekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train1_mergeDataset_add = train1_mergeDataset
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train4_Weekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)

print "make train4_Y .........."
train4_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*81, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*82, unit='D')))]
train4_noWeekend_Y = train4_Y[(train4_Y.day_of_week>=1) & (train4_Y.day_of_week<=5)]
train4_Weekend_Y = train4_Y[(train4_Y.day_of_week==6) | (train4_Y.day_of_week==7)]

final_train4_noWeekend = pd.merge(train4_noWeekend_mergeDataset_add,train4_noWeekend_Y,on=['user_id', 'day_of_week'], how='left')
final_train4_Weekend = pd.merge(train4_Weekend_mergeDataset_add,train4_Weekend_Y,on=['user_id', 'day_of_week'], how='left')


print "select test dataset ............."
test = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*70, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*82, unit='D')))]
test_noWeekend = test[(test.day_of_week>=1) & (test.day_of_week<=5)]
test_Weekend = test[(test.day_of_week==6) | (test.day_of_week==7)]

print "============ test_noWeekend ==========="
test_noWeekend_MeanStdSum = test_noWeekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
test_noWeekend_MeanStdSum['power_rate'] = test_noWeekend_MeanStdSum.power_sum / test_noWeekend_MeanStdSum.power_sum.sum()
test_noWeekend_DOW_MeanStdSum = test_noWeekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
test_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
test_noWeekend_DOW_MeanStdSumAllsum = test_noWeekend_DOW_MeanStdSum.merge(test_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
test_noWeekend_DOW_MeanStdSumAllsum['DOW_power_rate'] = test_noWeekend_DOW_MeanStdSumAllsum.DOW_power_sum / test_noWeekend_DOW_MeanStdSumAllsum.DOW_allsum
test_noWeekend_mergeDataset = pd.merge(test_noWeekend_DOW_MeanStdSumAllsum, test_noWeekend_MeanStdSum, on='user_id', how='left')

test_noWeekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(test_noWeekend)
test_noWeekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

test_noWeekend_mergeDataset_add = pd.merge(test_noWeekend_mergeDataset,test_noWeekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train1_mergeDataset_add = train1_mergeDataset
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
test_noWeekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)


print "============ test_Weekend ==========="
test_Weekend_MeanStdSum = test_Weekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
test_Weekend_MeanStdSum['power_rate'] = test_Weekend_MeanStdSum.power_sum / test_Weekend_MeanStdSum.power_sum.sum()
test_Weekend_DOW_MeanStdSum = test_Weekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
test_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
test_Weekend_DOW_MeanStdSumAllsum = test_Weekend_DOW_MeanStdSum.merge(test_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
test_Weekend_DOW_MeanStdSumAllsum['DOW_power_rate'] = test_Weekend_DOW_MeanStdSumAllsum.DOW_power_sum / test_Weekend_DOW_MeanStdSumAllsum.DOW_allsum
test_Weekend_mergeDataset = pd.merge(test_Weekend_DOW_MeanStdSumAllsum, test_Weekend_MeanStdSum, on='user_id', how='left')

test_Weekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(test_Weekend)
test_Weekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

test_Weekend_mergeDataset_add = pd.merge(test_Weekend_mergeDataset,test_Weekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train1_mergeDataset_add = train1_mergeDataset
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
test_Weekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)


test_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*82, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*83, unit='D')))]
test_noWeekend_Y = test_Y[(test_Y.day_of_week>=1) & (test_Y.day_of_week<=5)]
test_Weekend_Y = test_Y[(test_Y.day_of_week==6) | (test_Y.day_of_week==7)]

final_test_noWeekend = pd.merge(test_noWeekend_mergeDataset_add,test_noWeekend_Y,on=['user_id', 'day_of_week'], how='left')
final_test_Weekend = pd.merge(test_Weekend_mergeDataset_add,test_Weekend_Y,on=['user_id', 'day_of_week'], how='left')



print "make all train dataset ........."
print "make train_noWeekend dataset .........."
#train = pd.concat([final_train1, final_train2, final_train3, final_train4], axis=0, ignore_index=True)
train1_noWeekend_matrix = final_train1_noWeekend.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train2_noWeekend_matrix = final_train2_noWeekend.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train3_noWeekend_matrix = final_train3_noWeekend.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train4_noWeekend_matrix = final_train4_noWeekend.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
#final_train_matrix = train.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
final_train_noWeekend_matrix = np.row_stack((train1_noWeekend_matrix, train2_noWeekend_matrix, train3_noWeekend_matrix, train4_noWeekend_matrix))
train_noWeekend_X = final_train_noWeekend_matrix[:,:-1]
train_noWeekend_Y = final_train_noWeekend_matrix[:,-1]
print "make train_Weekend dataset .........."
#train = pd.concat([final_train1, final_train2, final_train3, final_train4], axis=0, ignore_index=True)
train1_Weekend_matrix = final_train1_Weekend.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train2_Weekend_matrix = final_train2_Weekend.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train3_Weekend_matrix = final_train3_Weekend.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train4_Weekend_matrix = final_train4_Weekend.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
#final_train_matrix = train.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
final_train_Weekend_matrix = np.row_stack((train1_Weekend_matrix, train2_Weekend_matrix, train3_Weekend_matrix, train4_Weekend_matrix))
train_Weekend_X = final_train_Weekend_matrix[:,:-1]
train_Weekend_Y = final_train_Weekend_matrix[:,-1]

print "make test datset .........."
print "make test_noWeekend dataset .........."
final_test_noWeekend_matrix = final_test_noWeekend.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
test_noWeekend_matrix_X = final_test_noWeekend_matrix[:,:-1]
test_noWeekend_matrix_Y = final_test_noWeekend_matrix[:,-1]
print "make test_Weekend dataset .........."
final_test_Weekend_matrix = final_test_Weekend.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
test_Weekend_matrix_X = final_test_Weekend_matrix[:,:-1]
test_Weekend_matrix_Y = final_test_Weekend_matrix[:,-1]

print "========================= noWeekend =============================="
print("noWeekend hyper-parameter optimization...................")
noWeekend_xgb_model = xgb.XGBRegressor()
noWeekend_params = {'max_depth':[2,3,4,5,6], 'learning_rate':[0.05,0.1,0.15], 'n_estimators':[50,100,150,200], 'max_delta_step':[1] ,
          'objective':['reg:linear', 'reg:gamma','reg:tweedie',]}
# , 'colsample_bytree':[1], 'colsample_bylevel':[1], 'reg_alpha':[0], 'reg_lambda':[1], 'scale_pos_weight':[1], 'base_score':[0.5], 'seed':[0], 'missing':[None],'nthread':[-1], 'gamma':[0], 'min_child_weight':[1], , 'subsample':[0.5,0.8,1]
noWeekend_gridsearchcvRegression = GridSearchCV(noWeekend_xgb_model, noWeekend_params, iid=True,scoring=None, n_jobs=1, refit=True, verbose=2, return_train_score=True)

print "noWeekend optimization fitting ..............."
noWeekend_gridsearchcvRegression.fit(train_noWeekend_X,train_noWeekend_Y)

print "/n"
print "noWeekend Best Score : ",noWeekend_gridsearchcvRegression.best_score_
print "noWeekend Best Params : ",noWeekend_gridsearchcvRegression.best_params_

print "noWeekend predict fiting............."
noWeekend_xgb_model = xgb.XGBRegressor(n_estimators=noWeekend_gridsearchcvRegression.best_params_['n_estimators'],max_depth=noWeekend_gridsearchcvRegression.best_params_['max_depth'],objective=noWeekend_gridsearchcvRegression.best_params_['objective'], max_delta_step=1, learning_rate=noWeekend_gridsearchcvRegression.best_params_['learning_rate'],silent=False)
noWeekend_xgb_model.fit(train_noWeekend_X, train_noWeekend_Y)

print "save model .........."
noWeekend_input_flag = raw_input("Do you want to save the model by train? Input[y/n]:")
if noWeekend_input_flag == 'y':
	joblib.dump(noWeekend_xgb_model, "../model/noWeekend_F1_xgb_model.m")

print "noWeekend predict ............"
predict_noWeekend_Y = noWeekend_xgb_model.predict(test_noWeekend_matrix_X)
predict_noWeekend_Y = np.round(predict_noWeekend_Y).astype(int)

print "noWeekend MSE = ",metrics.mean_squared_error(test_noWeekend_matrix_Y, predict_noWeekend_Y)
noWeekend_summ =0
for i in range(len(test_noWeekend_matrix_Y)):
    if (predict_noWeekend_Y[i]+test_noWeekend_matrix_Y[i]) == 0:
        continue
    noWeekend_summ+=abs((predict_noWeekend_Y[i]-test_noWeekend_matrix_Y[i])/(predict_noWeekend_Y[i]+test_noWeekend_matrix_Y[i]))
noWeekend_meansum = noWeekend_summ/len(test_noWeekend_matrix_Y)
print "noWeekend MeanSum = ",noWeekend_meansum
print "noWeekend corrcoef = ",np.corrcoef(predict_noWeekend_Y, test_noWeekend_matrix_Y)


print "========================= Weekend =============================="
print("Weekend hyper-parameter optimization...................")
Weekend_xgb_model = xgb.XGBRegressor()
Weekend_params = {'max_depth':[2,3,4,5,6], 'learning_rate':[0.05,0.1,0.15], 'n_estimators':[50,100,150,200], 'max_delta_step':[1] ,
          'objective':['reg:linear', 'reg:gamma','reg:tweedie',]}
# , 'colsample_bytree':[1], 'colsample_bylevel':[1], 'reg_alpha':[0], 'reg_lambda':[1], 'scale_pos_weight':[1], 'base_score':[0.5], 'seed':[0], 'missing':[None],'nthread':[-1], 'gamma':[0], 'min_child_weight':[1], , 'subsample':[0.5,0.8,1]
Weekend_gridsearchcvRegression = GridSearchCV(Weekend_xgb_model, Weekend_params, iid=True,scoring=None, n_jobs=1, refit=True, verbose=2, return_train_score=True)

print "Weekend optimization fitting ..............."
Weekend_gridsearchcvRegression.fit(train_Weekend_X,train_Weekend_Y)

print "/n"
print "Weekend Best Score : ",Weekend_gridsearchcvRegression.best_score_
print "Weekend Best Params : ",Weekend_gridsearchcvRegression.best_params_

print "Weekend predict fiting............."
Weekend_xgb_model = xgb.XGBRegressor(n_estimators=Weekend_gridsearchcvRegression.best_params_['n_estimators'],max_depth=Weekend_gridsearchcvRegression.best_params_['max_depth'],objective=Weekend_gridsearchcvRegression.best_params_['objective'], max_delta_step=1, learning_rate=Weekend_gridsearchcvRegression.best_params_['learning_rate'],silent=False)
Weekend_xgb_model.fit(train_Weekend_X, train_Weekend_Y)

print "save model ........."
Weekend_input_flag = raw_input("Do you want to save the model by train? Input[y/n]:")
if Weekend_input_flag == 'y':
	joblib.dump(Weekend_xgb_model, "../model/Weekend_F1_xgb_model.m")

print "Weekend predict ............"
predict_Weekend_Y = Weekend_xgb_model.predict(test_Weekend_matrix_X)
predict_Weekend_Y = np.round(predict_Weekend_Y).astype(int)

print "Weekend MSE = ",metrics.mean_squared_error(test_Weekend_matrix_Y, predict_Weekend_Y)
Weekend_summ =0
for i in range(len(test_Weekend_matrix_Y)):
    if (predict_Weekend_Y[i]+test_Weekend_matrix_Y[i]) == 0:
        continue
    Weekend_summ+=abs((predict_Weekend_Y[i]-test_Weekend_matrix_Y[i])/(predict_Weekend_Y[i]+test_Weekend_matrix_Y[i]))
Weekend_meansum = Weekend_summ/len(test_Weekend_matrix_Y)
print "Weekend MeanSum = ",Weekend_meansum
print "Weekend corrcoef = ",np.corrcoef(predict_Weekend_Y, test_Weekend_matrix_Y)


print "create next datasets ..........."
train1_noWeekend_matrix_X = train1_noWeekend_matrix[:,:-1]
train1_predict_noWeekend_Y = noWeekend_xgb_model.predict(train1_noWeekend_matrix_X)
train1_noWeekend_Y['power_consumption'] = np.round(train1_predict_noWeekend_Y).astype(int)
train1AndPredictY_noWeekend = pd.concat([train1_noWeekend, train1_noWeekend_Y], axis=0, ignore_index=True)
train1_Weekend_matrix_X = train1_Weekend_matrix[:,:-1]
train1_predict_Weekend_Y = Weekend_xgb_model.predict(train1_Weekend_matrix_X)
train1_Weekend_Y['power_consumption'] = np.round(train1_predict_Weekend_Y).astype(int)
train1AndPredictY_Weekend = pd.concat([train1_Weekend, train1_Weekend_Y], axis=0, ignore_index=True)
train1AndPredictY = pd.concat([train1AndPredictY_noWeekend, train1AndPredictY_Weekend], axis=0, ignore_index=True)
train1AndPredictY.to_csv(u'../F1_Result/train1AndPredictY.csv', header=True, index=False)

train2_noWeekend_matrix_X = train2_noWeekend_matrix[:,:-1]
train2_predict_noWeekend_Y = noWeekend_xgb_model.predict(train2_noWeekend_matrix_X)
train2_noWeekend_Y['power_consumption'] = np.round(train2_predict_noWeekend_Y).astype(int)
train2AndPredictY_noWeekend = pd.concat([train2_noWeekend, train2_noWeekend_Y], axis=0, ignore_index=True)
train2_Weekend_matrix_X = train2_Weekend_matrix[:,:-1]
train2_predict_Weekend_Y = Weekend_xgb_model.predict(train2_Weekend_matrix_X)
train2_Weekend_Y['power_consumption'] = np.round(train2_predict_Weekend_Y).astype(int)
train2AndPredictY_Weekend = pd.concat([train2_Weekend, train2_Weekend_Y], axis=0, ignore_index=True)
train2AndPredictY = pd.concat([train2AndPredictY_noWeekend, train2AndPredictY_Weekend], axis=0, ignore_index=True)
train2AndPredictY.to_csv(u'../F1_Result/train2AndPredictY.csv', header=True, index=False)

train3_noWeekend_matrix_X = train3_noWeekend_matrix[:,:-1]
train3_predict_noWeekend_Y = noWeekend_xgb_model.predict(train3_noWeekend_matrix_X)
train3_noWeekend_Y['power_consumption'] = np.round(train3_predict_noWeekend_Y).astype(int)
train3AndPredictY_noWeekend = pd.concat([train3_noWeekend, train3_noWeekend_Y], axis=0, ignore_index=True)
train3_Weekend_matrix_X = train3_Weekend_matrix[:,:-1]
train3_predict_Weekend_Y = Weekend_xgb_model.predict(train3_Weekend_matrix_X)
train3_Weekend_Y['power_consumption'] = np.round(train3_predict_Weekend_Y).astype(int)
train3AndPredictY_Weekend = pd.concat([train3_Weekend, train3_Weekend_Y], axis=0, ignore_index=True)
train3AndPredictY = pd.concat([train3AndPredictY_noWeekend, train3AndPredictY_Weekend], axis=0, ignore_index=True)
train3AndPredictY.to_csv(u'../F1_Result/train3AndPredictY.csv', header=True, index=False)

train4_noWeekend_matrix_X = train4_noWeekend_matrix[:,:-1]
train4_predict_noWeekend_Y = noWeekend_xgb_model.predict(train4_noWeekend_matrix_X)
train4_noWeekend_Y['power_consumption'] = np.round(train4_predict_noWeekend_Y).astype(int)
train4AndPredictY_noWeekend = pd.concat([train4_noWeekend, train4_noWeekend_Y], axis=0, ignore_index=True)
train4_Weekend_matrix_X = train4_Weekend_matrix[:,:-1]
train4_predict_Weekend_Y = Weekend_xgb_model.predict(train4_Weekend_matrix_X)
train4_Weekend_Y['power_consumption'] = np.round(train4_predict_Weekend_Y).astype(int)
train4AndPredictY_Weekend = pd.concat([train4_Weekend, train4_Weekend_Y], axis=0, ignore_index=True)
train4AndPredictY = pd.concat([train4AndPredictY_noWeekend, train4AndPredictY_Weekend], axis=0, ignore_index=True)
train4AndPredictY.to_csv(u'../F1_Result/train4AndPredictY.csv', header=True, index=False)

test_noWeekend_Y['power_consumption'] = np.round(predict_noWeekend_Y).astype(int)
test_Weekend_Y['power_consumption'] = np.round(predict_Weekend_Y).astype(int)
testAndPredictY_noWeekend = pd.concat([test_noWeekend, test_noWeekend_Y], axis=0, ignore_index=True)
testAndPredictY_Weekend = pd.concat([test_Weekend, test_Weekend_Y], axis=0, ignore_index=True)
testAndPredictY = pd.concat([testAndPredictY_noWeekend, testAndPredictY_Weekend], axis=0, ignore_index=True)
testAndPredictY.to_csv(u'../F1_Result/testAndPredictY.csv', header=True, index=False)
