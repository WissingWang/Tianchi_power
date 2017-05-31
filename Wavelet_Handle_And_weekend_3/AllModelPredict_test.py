import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.externals import joblib
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn import metrics
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import TransformByDOW
import AllModelWaveletTransformByDOW

print "loading model ..........."
noWeekend_F1_xgb_model = joblib.load("./model/noWeekend_F1_xgb_model.m")
noWeekend_F2_xgb_model = joblib.load("./model/noWeekend_F2_xgb_model.m")
noWeekend_F3_xgb_model = joblib.load("./model/noWeekend_F3_xgb_model.m")
noWeekend_F4_xgb_model = joblib.load("./model/noWeekend_F4_xgb_model.m")
noWeekend_F5_xgb_model = joblib.load("./model/noWeekend_F5_xgb_model.m")
Weekend_F1_xgb_model = joblib.load("./model/Weekend_F1_xgb_model.m")
Weekend_F2_xgb_model = joblib.load("./model/Weekend_F2_xgb_model.m")
Weekend_F3_xgb_model = joblib.load("./model/Weekend_F3_xgb_model.m")
Weekend_F4_xgb_model = joblib.load("./model/Weekend_F4_xgb_model.m")

print "create the part dataset of predicting ........."
rng = pd.date_range('7/28/2016', '8/26/2016')
predicteData = pd.DataFrame(rng, columns=['predict_date'])
def addDayOfWeek(x):
    return x.weekday() + 1
predicteData['day_of_week'] = predicteData.predict_date.apply(addDayOfWeek)

print "loading dataset .............."
handledataset = pd.read_csv(u'/home/haven/Tianchi_power/Tianchi_power__boxhandle15_DayOfWeek.csv')
groupbydataset = TransformByDOW.transformByDayOfWeek()
groupbydataset.drop('level_2', axis=1,inplace=True)

print "transform date to datetime .............."
handledataset.record_date = pd.to_datetime(handledataset.record_date)



print "select features about 1st week............."
features1 = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*70, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*82, unit='D')))]

features1_noWeekend = features1[(features1.day_of_week>=1) & (features1.day_of_week<=5)]
features1_Weekend = features1[(features1.day_of_week==6) | (features1.day_of_week==7)]

print "============ features1_noWeekend ==========="
features1_noWeekend_MeanStdSum = features1_noWeekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features1_noWeekend_MeanStdSum['power_rate'] = features1_noWeekend_MeanStdSum.power_sum / features1_noWeekend_MeanStdSum.power_sum.sum()
features1_noWeekend_DOW_MeanStdSum = features1_noWeekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features1_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features1_noWeekend_DOW_MeanStdSumAllsum = features1_noWeekend_DOW_MeanStdSum.merge(features1_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features1_noWeekend_DOW_MeanStdSumAllsum['DOW_powaer_rate'] = features1_noWeekend_DOW_MeanStdSumAllsum.DOW_power_sum / features1_noWeekend_DOW_MeanStdSumAllsum.DOW_allsum
features1_noWeekend_mergeDataset = pd.merge(features1_noWeekend_DOW_MeanStdSumAllsum, features1_noWeekend_MeanStdSum, on='user_id', how='left')

features1_noWeekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features1_noWeekend)
features1_noWeekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features1_noWeekend_mergeDataset_add = pd.merge(features1_noWeekend_mergeDataset, features1_noWeekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features1_noWeekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_powaer_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)

print "============ features1_Weekend ==========="
features1_Weekend_MeanStdSum = features1_Weekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features1_Weekend_MeanStdSum['power_rate'] = features1_Weekend_MeanStdSum.power_sum / features1_Weekend_MeanStdSum.power_sum.sum()
features1_Weekend_DOW_MeanStdSum = features1_Weekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features1_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features1_Weekend_DOW_MeanStdSumAllsum = features1_Weekend_DOW_MeanStdSum.merge(features1_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features1_Weekend_DOW_MeanStdSumAllsum['DOW_powaer_rate'] = features1_Weekend_DOW_MeanStdSumAllsum.DOW_power_sum / features1_Weekend_DOW_MeanStdSumAllsum.DOW_allsum
features1_Weekend_mergeDataset = pd.merge(features1_Weekend_DOW_MeanStdSumAllsum, features1_Weekend_MeanStdSum, on='user_id', how='left')

features1_Weekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features1_Weekend)
features1_Weekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features1_Weekend_mergeDataset_add = pd.merge(features1_Weekend_mergeDataset, features1_Weekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features1_Weekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_powaer_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)

features1_Y = predicteData[(predicteData.predict_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*82, unit='D'))) & (predicteData.predict_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*83, unit='D')))]
features1_noWeekend_Y = features1_Y[(features1_Y.day_of_week>=1) & (features1_Y.day_of_week<=5)]
features1_Weekend_Y = features1_Y[(features1_Y.day_of_week==6) | (features1_Y.day_of_week==7)]
final_features1_noWeekend = pd.merge(features1_noWeekend_mergeDataset_add,features1_noWeekend_Y,on='day_of_week', how='left')
final_features1_Weekend = pd.merge(features1_Weekend_mergeDataset_add,features1_Weekend_Y,on='day_of_week', how='left')

final_features1_noWeekend_Y = final_features1_noWeekend[['user_id', 'day_of_week', 'predict_date']]
final_features1_noWeekend_Y.rename(columns={"predict_date": "record_date"}, inplace=True)
final_features1_Weekend_Y = final_features1_Weekend[['user_id', 'day_of_week', 'predict_date']]
final_features1_Weekend_Y.rename(columns={"predict_date": "record_date"}, inplace=True)

print "predict 1st week ................."
print "predict noWeekend ................."
features1_noWeekend_matrix = final_features1_noWeekend.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_noWeekend_Y1 = noWeekend_F1_xgb_model.predict(features1_noWeekend_matrix)
final_features1_noWeekend['predict_power_consumption'] = predict_noWeekend_Y1
final_features1_noWeekend_Y['power_consumption'] = predict_noWeekend_Y1
features1AndPredictY_noWeekend = pd.concat([features1_noWeekend, final_features1_noWeekend_Y], axis=0, ignore_index=True)
print "predict Weekend ................."
features1_Weekend_matrix = final_features1_Weekend.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_Weekend_Y1 = Weekend_F1_xgb_model.predict(features1_Weekend_matrix)
final_features1_Weekend['predict_power_consumption'] = predict_Weekend_Y1
final_features1_Weekend_Y['power_consumption'] = predict_Weekend_Y1
features1AndPredictY_Weekend = pd.concat([features1_Weekend, final_features1_Weekend_Y], axis=0, ignore_index=True)

final_features1 = pd.concat([final_features1_noWeekend, final_features1_Weekend], axis=0, ignore_index=True)
features1AndPredictY = pd.concat([features1AndPredictY_noWeekend, features1AndPredictY_Weekend], axis=0, ignore_index=True)
features1AndPredictY = features1AndPredictY[features1AndPredictY.record_date >= np.min(features1AndPredictY.record_date)+pd.to_timedelta(7, unit='D')]


print "select features about 2st week............."
features2 = features1AndPredictY

features2_noWeekend = features2[(features2.day_of_week>=1) & (features2.day_of_week<=5)]
features2_Weekend = features2[(features2.day_of_week==6) | (features2.day_of_week==7)]

print "============ features2_noWeekend ==========="
features2_noWeekend_MeanStdSum = features2_noWeekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features2_noWeekend_MeanStdSum['power_rate'] = features2_noWeekend_MeanStdSum.power_sum / features2_noWeekend_MeanStdSum.power_sum.sum()
features2_noWeekend_DOW_MeanStdSum = features2_noWeekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features2_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features2_noWeekend_DOW_MeanStdSumAllsum = features2_noWeekend_DOW_MeanStdSum.merge(features2_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features2_noWeekend_DOW_MeanStdSumAllsum['DOW_powaer_rate'] = features2_noWeekend_DOW_MeanStdSumAllsum.DOW_power_sum / features2_noWeekend_DOW_MeanStdSumAllsum.DOW_allsum
features2_noWeekend_mergeDataset = pd.merge(features2_noWeekend_DOW_MeanStdSumAllsum, features2_noWeekend_MeanStdSum, on='user_id', how='left')

features2_noWeekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features2_noWeekend)
features2_noWeekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features2_noWeekend_mergeDataset_add = pd.merge(features2_noWeekend_mergeDataset, features2_noWeekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features2_noWeekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_powaer_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)


print "============ features2_Weekend ==========="
features2_Weekend_MeanStdSum = features2_Weekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features2_Weekend_MeanStdSum['power_rate'] = features2_Weekend_MeanStdSum.power_sum / features2_Weekend_MeanStdSum.power_sum.sum()
features2_Weekend_DOW_MeanStdSum = features2_Weekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features2_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features2_Weekend_DOW_MeanStdSumAllsum = features2_Weekend_DOW_MeanStdSum.merge(features2_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features2_Weekend_DOW_MeanStdSumAllsum['DOW_powaer_rate'] = features2_Weekend_DOW_MeanStdSumAllsum.DOW_power_sum / features2_Weekend_DOW_MeanStdSumAllsum.DOW_allsum
features2_Weekend_mergeDataset = pd.merge(features2_Weekend_DOW_MeanStdSumAllsum, features2_Weekend_MeanStdSum, on='user_id', how='left')

features2_Weekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features2_Weekend)
features2_Weekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features2_Weekend_mergeDataset_add = pd.merge(features2_Weekend_mergeDataset, features2_Weekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features2_Weekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_powaer_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)


features2_Y = predicteData[(predicteData.predict_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*83, unit='D'))) & (predicteData.predict_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*84, unit='D')))]
features2_noWeekend_Y = features2_Y[(features2_Y.day_of_week>=1) & (features2_Y.day_of_week<=5)]
features2_Weekend_Y = features2_Y[(features2_Y.day_of_week==6) | (features2_Y.day_of_week==7)]
final_features2_noWeekend = pd.merge(features2_noWeekend_mergeDataset_add,features2_noWeekend_Y,on='day_of_week', how='left')
final_features2_Weekend = pd.merge(features2_Weekend_mergeDataset_add,features2_Weekend_Y,on='day_of_week', how='left')

final_features2_noWeekend_Y = final_features2_noWeekend[['user_id', 'day_of_week', 'predict_date']]
final_features2_noWeekend_Y.rename(columns={"predict_date": "record_date"}, inplace=True)
final_features2_Weekend_Y = final_features2_Weekend[['user_id', 'day_of_week', 'predict_date']]
final_features2_Weekend_Y.rename(columns={"predict_date": "record_date"}, inplace=True)


print "predict 2st week ................."
print "predict noWeekend ................."
features2_noWeekend_matrix = final_features2_noWeekend.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_noWeekend_Y2 = noWeekend_F2_xgb_model.predict(features2_noWeekend_matrix)
final_features2_noWeekend['predict_power_consumption'] = predict_noWeekend_Y2
final_features2_noWeekend_Y['power_consumption'] = predict_noWeekend_Y2
features2AndPredictY_noWeekend = pd.concat([features2_noWeekend, final_features2_noWeekend_Y], axis=0, ignore_index=True)
print "predict Weekend ................."
features2_Weekend_matrix = final_features2_Weekend.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_Weekend_Y2 = Weekend_F2_xgb_model.predict(features2_Weekend_matrix)
final_features2_Weekend['predict_power_consumption'] = predict_Weekend_Y2
final_features2_Weekend_Y['power_consumption'] = predict_Weekend_Y2
features2AndPredictY_Weekend = pd.concat([features2_Weekend, final_features2_Weekend_Y], axis=0, ignore_index=True)

final_features2 = pd.concat([final_features2_noWeekend, final_features2_Weekend], axis=0, ignore_index=True)
features2AndPredictY = pd.concat([features2AndPredictY_noWeekend, features2AndPredictY_Weekend], axis=0, ignore_index=True)
features2AndPredictY = features2AndPredictY[features2AndPredictY.record_date >= np.min(features2AndPredictY.record_date)+pd.to_timedelta(7, unit='D')]


print "select features about 3st week............."
features3 = features2AndPredictY

features3_noWeekend = features3[(features3.day_of_week>=1) & (features3.day_of_week<=5)]
features3_Weekend = features3[(features3.day_of_week==6) | (features3.day_of_week==7)]

print "============ features3_noWeekend ==========="
features3_noWeekend_MeanStdSum = features3_noWeekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features3_noWeekend_MeanStdSum['power_rate'] = features3_noWeekend_MeanStdSum.power_sum / features3_noWeekend_MeanStdSum.power_sum.sum()
features3_noWeekend_DOW_MeanStdSum = features3_noWeekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features3_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features3_noWeekend_DOW_MeanStdSumAllsum = features3_noWeekend_DOW_MeanStdSum.merge(features3_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features3_noWeekend_DOW_MeanStdSumAllsum['DOW_powaer_rate'] = features3_noWeekend_DOW_MeanStdSumAllsum.DOW_power_sum / features3_noWeekend_DOW_MeanStdSumAllsum.DOW_allsum
features3_noWeekend_mergeDataset = pd.merge(features3_noWeekend_DOW_MeanStdSumAllsum, features3_noWeekend_MeanStdSum, on='user_id', how='left')

features3_noWeekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features3_noWeekend)
features3_noWeekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features3_noWeekend_mergeDataset_add = pd.merge(features3_noWeekend_mergeDataset, features3_noWeekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features3_noWeekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_powaer_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)

print "============ features3_Weekend ==========="
features3_Weekend_MeanStdSum = features3_Weekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features3_Weekend_MeanStdSum['power_rate'] = features3_Weekend_MeanStdSum.power_sum / features3_Weekend_MeanStdSum.power_sum.sum()
features3_Weekend_DOW_MeanStdSum = features3_Weekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features3_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features3_Weekend_DOW_MeanStdSumAllsum = features3_Weekend_DOW_MeanStdSum.merge(features3_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features3_Weekend_DOW_MeanStdSumAllsum['DOW_powaer_rate'] = features3_Weekend_DOW_MeanStdSumAllsum.DOW_power_sum / features3_Weekend_DOW_MeanStdSumAllsum.DOW_allsum
features3_Weekend_mergeDataset = pd.merge(features3_Weekend_DOW_MeanStdSumAllsum, features3_Weekend_MeanStdSum, on='user_id', how='left')

features3_Weekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features3_Weekend)
features3_Weekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features3_Weekend_mergeDataset_add = pd.merge(features3_Weekend_mergeDataset, features3_Weekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features3_Weekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_powaer_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)


features3_Y = predicteData[(predicteData.predict_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*84, unit='D'))) & (predicteData.predict_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*85, unit='D')))]
features3_noWeekend_Y = features3_Y[(features3_Y.day_of_week>=1) & (features3_Y.day_of_week<=5)]
features3_Weekend_Y = features3_Y[(features3_Y.day_of_week==6) | (features3_Y.day_of_week==7)]
final_features3_noWeekend = pd.merge(features3_noWeekend_mergeDataset_add,features3_noWeekend_Y,on='day_of_week', how='left')
final_features3_Weekend = pd.merge(features3_Weekend_mergeDataset_add,features3_Weekend_Y,on='day_of_week', how='left')

final_features3_noWeekend_Y = final_features3_noWeekend[['user_id', 'day_of_week', 'predict_date']]
final_features3_noWeekend_Y.rename(columns={"predict_date": "record_date"}, inplace=True)
final_features3_Weekend_Y = final_features3_Weekend[['user_id', 'day_of_week', 'predict_date']]
final_features3_Weekend_Y.rename(columns={"predict_date": "record_date"}, inplace=True)

print "predict 3st week ................."
print "predict noWeekend ................."
features3_noWeekend_matrix = final_features3_noWeekend.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_noWeekend_Y3 = noWeekend_F3_xgb_model.predict(features3_noWeekend_matrix)
final_features3_noWeekend['predict_power_consumption'] = predict_noWeekend_Y3
final_features3_noWeekend_Y['power_consumption'] = predict_noWeekend_Y3
features3AndPredictY_noWeekend = pd.concat([features3_noWeekend, final_features3_noWeekend_Y], axis=0, ignore_index=True)
print "predict Weekend ................."
features3_Weekend_matrix = final_features3_Weekend.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_Weekend_Y3 = Weekend_F3_xgb_model.predict(features3_Weekend_matrix)
final_features3_Weekend['predict_power_consumption'] = predict_Weekend_Y3
final_features3_Weekend_Y['power_consumption'] = predict_Weekend_Y3
features3AndPredictY_Weekend = pd.concat([features3_Weekend, final_features3_Weekend_Y], axis=0, ignore_index=True)

final_features3 = pd.concat([final_features3_noWeekend, final_features3_Weekend], axis=0, ignore_index=True)
features3AndPredictY = pd.concat([features3AndPredictY_noWeekend, features3AndPredictY_Weekend], axis=0, ignore_index=True)
features3AndPredictY = features3AndPredictY[features3AndPredictY.record_date >= np.min(features3AndPredictY.record_date)+pd.to_timedelta(7, unit='D')]


print "select features about 4st week............."
features4 = features3AndPredictY

features4_noWeekend = features4[(features4.day_of_week>=1) & (features4.day_of_week<=5)]
features4_Weekend = features4[(features4.day_of_week==6) | (features4.day_of_week==7)]

print "============ features4_noWeekend ==========="
features4_noWeekend_MeanStdSum = features4_noWeekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features4_noWeekend_MeanStdSum['power_rate'] = features4_noWeekend_MeanStdSum.power_sum / features4_noWeekend_MeanStdSum.power_sum.sum()
features4_noWeekend_DOW_MeanStdSum = features4_noWeekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features4_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features4_noWeekend_DOW_MeanStdSumAllsum = features4_noWeekend_DOW_MeanStdSum.merge(features4_noWeekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features4_noWeekend_DOW_MeanStdSumAllsum['DOW_powaer_rate'] = features4_noWeekend_DOW_MeanStdSumAllsum.DOW_power_sum / features4_noWeekend_DOW_MeanStdSumAllsum.DOW_allsum
features4_noWeekend_mergeDataset = pd.merge(features4_noWeekend_DOW_MeanStdSumAllsum, features4_noWeekend_MeanStdSum, on='user_id', how='left')

features4_noWeekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features4_noWeekend)
features4_noWeekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features4_noWeekend_mergeDataset_add = pd.merge(features4_noWeekend_mergeDataset, features4_noWeekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features4_noWeekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_powaer_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)

print "============ features4_Weekend ==========="
features4_Weekend_MeanStdSum = features4_Weekend.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features4_Weekend_MeanStdSum['power_rate'] = features4_Weekend_MeanStdSum.power_sum / features4_Weekend_MeanStdSum.power_sum.sum()
features4_Weekend_DOW_MeanStdSum = features4_Weekend.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features4_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features4_Weekend_DOW_MeanStdSumAllsum = features4_Weekend_DOW_MeanStdSum.merge(features4_Weekend_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features4_Weekend_DOW_MeanStdSumAllsum['DOW_powaer_rate'] = features4_Weekend_DOW_MeanStdSumAllsum.DOW_power_sum / features4_Weekend_DOW_MeanStdSumAllsum.DOW_allsum
features4_Weekend_mergeDataset = pd.merge(features4_Weekend_DOW_MeanStdSumAllsum, features4_Weekend_MeanStdSum, on='user_id', how='left')

features4_Weekend_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features4_Weekend)
features4_Weekend_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features4_Weekend_mergeDataset_add = pd.merge(features4_Weekend_mergeDataset, features4_Weekend_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features4_Weekend_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_powaer_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)

features4_Y = predicteData[(predicteData.predict_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*85, unit='D'))) & (predicteData.predict_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*86, unit='D')))]
features4_noWeekend_Y = features4_Y[(features4_Y.day_of_week>=1) & (features4_Y.day_of_week<=5)]
features4_Weekend_Y = features4_Y[(features4_Y.day_of_week==6) | (features4_Y.day_of_week==7)]
final_features4_noWeekend = pd.merge(features4_noWeekend_mergeDataset_add,features4_noWeekend_Y,on='day_of_week', how='left')
final_features4_Weekend = pd.merge(features4_Weekend_mergeDataset_add,features4_Weekend_Y,on='day_of_week', how='left')

final_features4_noWeekend_Y = final_features4_noWeekend[['user_id', 'day_of_week', 'predict_date']]
final_features4_noWeekend_Y.rename(columns={"predict_date": "record_date"}, inplace=True)
final_features4_Weekend_Y = final_features4_Weekend[['user_id', 'day_of_week', 'predict_date']]
final_features4_Weekend_Y.rename(columns={"predict_date": "record_date"}, inplace=True)

print "predict 4st week ................."
print "predict noWeekend ................."
features4_noWeekend_matrix = final_features4_noWeekend.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_noWeekend_Y4 = noWeekend_F4_xgb_model.predict(features4_noWeekend_matrix)
final_features4_noWeekend['predict_power_consumption'] = predict_noWeekend_Y4
final_features4_noWeekend_Y['power_consumption'] = predict_noWeekend_Y4
features4AndPredictY_noWeekend = pd.concat([features4_noWeekend, final_features4_noWeekend_Y], axis=0, ignore_index=True)
print "predict Weekend ................."
features4_Weekend_matrix = final_features4_Weekend.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_Weekend_Y4 = Weekend_F4_xgb_model.predict(features4_Weekend_matrix)
final_features4_Weekend['predict_power_consumption'] = predict_Weekend_Y4
final_features4_Weekend_Y['power_consumption'] = predict_Weekend_Y4
features4AndPredictY_Weekend = pd.concat([features4_Weekend, final_features4_Weekend_Y], axis=0, ignore_index=True)

final_features4 = pd.concat([final_features4_noWeekend, final_features4_Weekend], axis=0, ignore_index=True)
features4AndPredictY = pd.concat([features4AndPredictY_noWeekend, features4AndPredictY_Weekend], axis=0, ignore_index=True)
features4AndPredictY = features4AndPredictY[features4AndPredictY.record_date >= np.min(features4AndPredictY.record_date)+pd.to_timedelta(7, unit='D')]

print "select features about 5st week............."
features5 = features4AndPredictY
features5_MeanStdSum = features5.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features5_MeanStdSum['power_rate'] = features5_MeanStdSum.power_sum / features5_MeanStdSum.power_sum.sum()
features5_DOW_MeanStdSum = features5.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features5_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features5_DOW_MeanStdSumAllsum = features5_DOW_MeanStdSum.merge(features5_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features5_DOW_MeanStdSumAllsum['DOW_powaer_rate'] = features5_DOW_MeanStdSumAllsum.DOW_power_sum / features5_DOW_MeanStdSumAllsum.DOW_allsum
features5_mergeDataset = pd.merge(features5_DOW_MeanStdSumAllsum, features5_MeanStdSum, on='user_id', how='left')

features5_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features5)
features5_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features5_mergeDataset_add = pd.merge(features5_mergeDataset,features5_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features5_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_powaer_rate', 'power_rate', 'power_mean', 'power_std'], axis=1,inplace=True)

features5_Y = predicteData[(predicteData.predict_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*86, unit='D'))) & (predicteData.predict_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*86+2, unit='D')))]
final_features5 = pd.merge(features5_mergeDataset_add,features5_Y,on='day_of_week', how='left')
final_features5.dropna(inplace=True)

print "predict 5st week ................."
features5_matrix = final_features5.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_Y5 = noWeekend_F5_xgb_model.predict(features5_matrix)
final_features5['predict_power_consumption'] = predict_Y5


#print "make all features dataset ........."
#features1_matrix = final_features1.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
#features2_matrix = final_features2.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
#features3_matrix = final_features3.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
#features4_matrix = final_features4.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
#features5_matrix = final_features5.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()


print "groupby for final_needdataset ................."
final_needdataset1 = final_features1.groupby('predict_date')['predict_power_consumption'].agg(sum).reset_index()
final_needdataset2 = final_features2.groupby('predict_date')['predict_power_consumption'].agg(sum).reset_index()
final_needdataset3 = final_features3.groupby('predict_date')['predict_power_consumption'].agg(sum).reset_index()
final_needdataset4 = final_features4.groupby('predict_date')['predict_power_consumption'].agg(sum).reset_index()
final_needdataset5 = final_features5.groupby('predict_date')['predict_power_consumption'].agg(sum).reset_index()

print "concat all final_needdataset*  (1--5) ................"
final_needdataset = pd.concat([final_needdataset1, final_needdataset2, final_needdataset3, final_needdataset4, final_needdataset5], axis=0, ignore_index=True)

print "save final_needdataset to Tianchi_power_predict_table.csv ............."
final_needdataset.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle_And_weekend_3/result_CSV/AllModel_Tianchi_power_predict_table_test.csv', header=True, index=False)







