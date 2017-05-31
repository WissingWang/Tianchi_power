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

print "loading model ..........."
F1_xgb_model = joblib.load("F1_xgb_model.m")

print "loading dataset .............." 
handledataset = pd.read_csv(u'/home/haven/Tianchi_power/Tianchi_power__boxhandle15_DayOfWeek.csv')

print "transform date to datetime .............."
handledataset.record_date = pd.to_datetime(handledataset.record_date)

print "select train1 dataset ............."
train1 = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*66, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*78, unit='D')))]
train1_MeanStdSum = train1.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train1_MeanStdSum['power_rate'] = train1_MeanStdSum.power_sum / train1_MeanStdSum.power_sum.sum()
train1_DOW_MeanStdSum = train1.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train1_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train1_DOW_MeanStdSumAllsum = train1_DOW_MeanStdSum.merge(train1_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train1_DOW_MeanStdSumAllsum['DOW_power_rate'] = train1_DOW_MeanStdSumAllsum.DOW_power_sum / train1_DOW_MeanStdSumAllsum.DOW_allsum
train1_mergeDataset = pd.merge(train1_DOW_MeanStdSumAllsum, train1_MeanStdSum, on='user_id', how='left')

train1_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train1)
train1_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train1_mergeDataset_add = pd.merge(train1_mergeDataset,train1_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train1_mergeDataset_add = train1_mergeDataset
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate'], axis=1,inplace=True)
train1_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*78, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*79, unit='D')))]
final_train1 = pd.merge(train1_mergeDataset_add,train1_Y,on=['user_id', 'day_of_week'], how='left')



print "select train2 dataset ............."
train2 = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*67, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*79, unit='D')))]
train2_MeanStdSum = train2.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train2_MeanStdSum['power_rate'] = train2_MeanStdSum.power_sum / train2_MeanStdSum.power_sum.sum()
train2_DOW_MeanStdSum = train2.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train2_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train2_DOW_MeanStdSumAllsum = train2_DOW_MeanStdSum.merge(train2_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train2_DOW_MeanStdSumAllsum['DOW_power_rate'] = train2_DOW_MeanStdSumAllsum.DOW_power_sum / train2_DOW_MeanStdSumAllsum.DOW_allsum
train2_mergeDataset = pd.merge(train2_DOW_MeanStdSumAllsum, train2_MeanStdSum, on='user_id', how='left')

train2_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train2)
train2_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train2_mergeDataset_add = pd.merge(train2_mergeDataset, train2_dayofweek_dataset, on=['user_id', 'day_of_week'], how='left')
#train2_mergeDataset_add = train2_mergeDataset
#train2_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train2_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train2_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train2_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate'], axis=1,inplace=True)

train2_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*79, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*80, unit='D')))]

final_train2 = pd.merge(train2_mergeDataset_add,train2_Y,on=['user_id', 'day_of_week'], how='left')




print "select train3 dataset ............."
train3 = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*68, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*80, unit='D')))]
train3_MeanStdSum = train3.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train3_MeanStdSum['power_rate'] = train3_MeanStdSum.power_sum / train3_MeanStdSum.power_sum.sum()
train3_DOW_MeanStdSum = train3.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train3_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train3_DOW_MeanStdSumAllsum = train3_DOW_MeanStdSum.merge(train3_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train3_DOW_MeanStdSumAllsum['DOW_power_rate'] = train3_DOW_MeanStdSumAllsum.DOW_power_sum / train3_DOW_MeanStdSumAllsum.DOW_allsum
train3_mergeDataset = pd.merge(train3_DOW_MeanStdSumAllsum, train3_MeanStdSum, on='user_id', how='left')

train3_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train3)
train3_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train3_mergeDataset_add = pd.merge(train3_mergeDataset,train3_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train3_mergeDataset_add = train3_mergeDataset
#train3_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train3_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train3_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train3_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate'], axis=1,inplace=True)

train3_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*80, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*81, unit='D')))]

final_train3 = pd.merge(train3_mergeDataset_add,train3_Y,on=['user_id', 'day_of_week'], how='left')




print "select train4 dataset ............."
train4 = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*69, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*81, unit='D')))]
train4_MeanStdSum = train4.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
train4_MeanStdSum['power_rate'] = train4_MeanStdSum.power_sum / train4_MeanStdSum.power_sum.sum()
train4_DOW_MeanStdSum = train4.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
train4_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
train4_DOW_MeanStdSumAllsum = train4_DOW_MeanStdSum.merge(train4_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
train4_DOW_MeanStdSumAllsum['DOW_power_rate'] = train4_DOW_MeanStdSumAllsum.DOW_power_sum / train4_DOW_MeanStdSumAllsum.DOW_allsum
train4_mergeDataset = pd.merge(train4_DOW_MeanStdSumAllsum, train4_MeanStdSum, on='user_id', how='left')

train4_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(train4)
train4_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

train4_mergeDataset_add = pd.merge(train4_mergeDataset,train4_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#train4_mergeDataset_add = train4_mergeDataset
#train4_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#train4_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#train4_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
train4_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate'], axis=1,inplace=True)

train4_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*81, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*82, unit='D')))]

final_train4 = pd.merge(train4_mergeDataset_add,train4_Y,on=['user_id', 'day_of_week'], how='left')


print "select test dataset ............."
test = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*70, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*82, unit='D')))]
test_MeanStdSum = train4.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
test_MeanStdSum['power_rate'] = test_MeanStdSum.power_sum / test_MeanStdSum.power_sum.sum()
test_DOW_MeanStdSum = test.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
test_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
test_DOW_MeanStdSumAllsum = test_DOW_MeanStdSum.merge(test_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
test_DOW_MeanStdSumAllsum['DOW_power_rate'] = test_DOW_MeanStdSumAllsum.DOW_power_sum / test_DOW_MeanStdSumAllsum.DOW_allsum
test_mergeDataset = pd.merge(test_DOW_MeanStdSumAllsum, test_MeanStdSum, on='user_id', how='left')

test_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(test)
test_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

test_mergeDataset_add = pd.merge(test_mergeDataset,test_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate'], axis=1,inplace=True)

test_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*82, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*83, unit='D')))]

final_test = pd.merge(test_mergeDataset_add,test_Y,on=['user_id', 'day_of_week'], how='left')



print "make all train dataset ........."
#train = pd.concat([final_train1, final_train2, final_train3, final_train4], axis=0, ignore_index=True)
train1_matrix = final_train1.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train2_matrix = final_train2.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train3_matrix = final_train3.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train4_matrix = final_train4.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()

print "make test datset"
final_test_matrix = final_test.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
test_matrix_X = final_test_matrix[:,:-1]
test_matrix_Y = final_test_matrix[:,-1]


print "create next datasets ..........."
train1_matrix_X = train1_matrix[:,:-1]
train1_predict_Y = F1_xgb_model.predict(train1_matrix_X)
train1_Y.drop(['power_consumption'], axis=1, inplace=True)
print train1_Y.shape
train1_Y['power_consumption'] = train1_predict_Y
print train1_Y.shape
#print len(train1)
train1AndPredictY = pd.concat([train1, train1_Y], axis=0, ignore_index=True)
#print len(train1AndPredictY)
train1AndPredictY.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle/F1_Result/train1AndPredictY.csv', header=True, index=False)

train2_matrix_X = train2_matrix[:,:-1]
train2_predict_Y = F1_xgb_model.predict(train2_matrix_X)
train2_Y.drop(['power_consumption'], axis=1, inplace=True)
print train2_Y.shape
train2_Y['power_consumption'] = train2_predict_Y
print train2_Y.shape
train2AndPredictY = pd.concat([train2, train2_Y], axis=0, ignore_index=True)
train2AndPredictY.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle/F1_Result/train2AndPredictY.csv', header=True, index=False)

train3_matrix_X = train3_matrix[:,:-1]
train3_predict_Y = F1_xgb_model.predict(train3_matrix_X)
train3_Y.drop(['power_consumption'], axis=1, inplace=True)
print train3_Y.shape
train3_Y['power_consumption'] = train3_predict_Y
print train3_Y.shape
train3AndPredictY = pd.concat([train3, train3_Y], axis=0, ignore_index=True)
train3AndPredictY.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle/F1_Result/train3AndPredictY.csv', header=True, index=False)

print "train4_matrix ........."
train4_matrix_X = train4_matrix[:,:-1]
train4_predict_Y = F1_xgb_model.predict(train4_matrix_X)
train4_Y.drop(['power_consumption'], axis=1, inplace=True)
print train4_Y.shape
train4_Y['power_consumption'] = train4_predict_Y
print train4_Y.shape
train4AndPredictY = pd.concat([train4, train4_Y], axis=0, ignore_index=True)
train4AndPredictY.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle/F1_Result/train4AndPredictY.csv', header=True, index=False)

print "test_matrix ........."
predict_Y = F1_xgb_model.predict(test_matrix_X)
test_Y.drop(['power_consumption'], axis=1, inplace=True)
print test_Y.shape
test_Y['power_consumption'] = predict_Y
print test_Y.shape
print len(test)
testAndPredictY = pd.concat([test, test_Y], axis=0, ignore_index=True)
print len(testAndPredictY)
testAndPredictY.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle/F1_Result/testAndPredictY.csv', header=True, index=False)
