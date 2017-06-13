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
F1_xgb_model = joblib.load("../model/F1_xgb_model.m")
F2_xgb_model = joblib.load("../model/F2_xgb_model.m")
F3_xgb_model = joblib.load("../model/F3_xgb_model.m")
F4_xgb_model = joblib.load("../model/F4_xgb_model.m")
F5_xgb_model = joblib.load("../model/F5_xgb_model.m")

print "create the part dataset of predicting ........."
rng = pd.date_range('9/1/2016', '9/30/2016')
predicteData = pd.DataFrame(rng, columns=['predict_date'])
def addDayOfWeek(x):
    return x.weekday() +1
predicteData['day_of_week'] = predicteData.predict_date.apply(addDayOfWeek)

print "loading dataset .............." 
handledataset = pd.read_csv(u'/home/haven/Tianchi_power/Tianchi_power_Hanle_addDayOfWeek.csv')
groupbydataset = TransformByDOW.transformByDayOfWeek()
groupbydataset.drop('level_2', axis=1,inplace=True)

print "transform date to datetime .............."
handledataset.record_date = pd.to_datetime(handledataset.record_date)



print "select features about 1st week............."
features1 = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*75, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*87, unit='D')))]
features1_MeanStdSum = features1.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features1_MeanStdSum['power_rate'] = features1_MeanStdSum.power_sum / features1_MeanStdSum.power_sum.sum()
features1_DOW_MeanStdSum = features1.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features1_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features1_DOW_MeanStdSumAllsum = features1_DOW_MeanStdSum.merge(features1_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features1_DOW_MeanStdSumAllsum['DOW_power_rate'] = features1_DOW_MeanStdSumAllsum.DOW_power_sum / features1_DOW_MeanStdSumAllsum.DOW_allsum
features1_mergeDataset = pd.merge(features1_DOW_MeanStdSumAllsum, features1_MeanStdSum, on='user_id', how='left')

features1_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features1)
features1_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features1_mergeDataset_add = pd.merge(features1_mergeDataset, features1_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features1_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_rate', 'power_std', 'DOW_power_mean', 'DOW_power_std'], axis=1,inplace=True)

features1_Y = predicteData[(predicteData.predict_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*87, unit='D'))) & (predicteData.predict_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*88, unit='D')))]
final_features1 = pd.merge(features1_mergeDataset_add,features1_Y,on='day_of_week', how='left')

final_features1_Y = final_features1[['user_id', 'day_of_week', 'predict_date']]
final_features1_Y.rename(columns={"predict_date": "record_date"}, inplace=True)

print "predict 1st week ................."
features1_matrix = final_features1.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_Y1 = F1_xgb_model.predict(features1_matrix)
print len(predict_Y1)
print len(features1_Y)
final_features1['predict_power_consumption'] = predict_Y1
final_features1_Y['power_consumption'] = predict_Y1
features1AndPredictY = pd.concat([features1, final_features1_Y], axis=0, ignore_index=True)




print "select features about 2st week............."
features2 = features1AndPredictY
features2_MeanStdSum = features2.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features2_MeanStdSum['power_rate'] = features2_MeanStdSum.power_sum / features2_MeanStdSum.power_sum.sum()
features2_DOW_MeanStdSum = features2.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features2_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features2_DOW_MeanStdSumAllsum = features2_DOW_MeanStdSum.merge(features2_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features2_DOW_MeanStdSumAllsum['DOW_power_rate'] = features2_DOW_MeanStdSumAllsum.DOW_power_sum / features2_DOW_MeanStdSumAllsum.DOW_allsum
features2_mergeDataset = pd.merge(features2_DOW_MeanStdSumAllsum, features2_MeanStdSum, on='user_id', how='left')

features2_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features2)
features2_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features2_mergeDataset_add = pd.merge(features2_mergeDataset,features2_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features2_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_rate', 'power_std', 'DOW_power_mean', 'DOW_power_std'], axis=1,inplace=True)

features2_Y = predicteData[(predicteData.predict_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*88, unit='D'))) & (predicteData.predict_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*89, unit='D')))]
final_features2 = pd.merge(features2_mergeDataset_add,features2_Y,on='day_of_week', how='left')

final_features2_Y = final_features2[['user_id', 'day_of_week', 'predict_date']]
final_features2_Y.rename(columns={"predict_date": "record_date"}, inplace=True)

print "predict 2st week ................."
features2_matrix = final_features2.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_Y2 = F2_xgb_model.predict(features2_matrix)
final_features2['predict_power_consumption'] = predict_Y2
final_features2_Y['power_consumption'] = predict_Y2
features2AndPredictY = pd.concat([features2, final_features2_Y], axis=0, ignore_index=True)



print "select features about 3st week............."
features3 = features2AndPredictY
features3_MeanStdSum = features3.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features3_MeanStdSum['power_rate'] = features3_MeanStdSum.power_sum / features3_MeanStdSum.power_sum.sum()
features3_DOW_MeanStdSum = features3.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features3_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features3_DOW_MeanStdSumAllsum = features3_DOW_MeanStdSum.merge(features3_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features3_DOW_MeanStdSumAllsum['DOW_power_rate'] = features3_DOW_MeanStdSumAllsum.DOW_power_sum / features3_DOW_MeanStdSumAllsum.DOW_allsum
features3_mergeDataset = pd.merge(features3_DOW_MeanStdSumAllsum, features3_MeanStdSum, on='user_id', how='left')

features3_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features3)
features3_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features3_mergeDataset_add = pd.merge(features3_mergeDataset,features3_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features3_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_rate', 'power_std', 'DOW_power_mean', 'DOW_power_std'], axis=1,inplace=True)

features3_Y = predicteData[(predicteData.predict_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*89, unit='D'))) & (predicteData.predict_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*90, unit='D')))]
final_features3 = pd.merge(features3_mergeDataset_add,features3_Y,on='day_of_week', how='left')

final_features3_Y = final_features3[['user_id', 'day_of_week', 'predict_date']]
final_features3_Y.rename(columns={"predict_date": "record_date"}, inplace=True)

print "predict 3st week ................."
features3_matrix = final_features3.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_Y3 = F3_xgb_model.predict(features3_matrix)
final_features3['predict_power_consumption'] = predict_Y3
final_features3_Y['power_consumption'] = predict_Y3
features3AndPredictY = pd.concat([features3, final_features3_Y], axis=0, ignore_index=True)



print "select features about 4st week............."
features4 = features3AndPredictY
features4_MeanStdSum = features4.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features4_MeanStdSum['power_rate'] = features4_MeanStdSum.power_sum / features4_MeanStdSum.power_sum.sum()
features4_DOW_MeanStdSum = features4.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features4_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features4_DOW_MeanStdSumAllsum = features4_DOW_MeanStdSum.merge(features4_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features4_DOW_MeanStdSumAllsum['DOW_power_rate'] = features4_DOW_MeanStdSumAllsum.DOW_power_sum / features4_DOW_MeanStdSumAllsum.DOW_allsum
features4_mergeDataset = pd.merge(features4_DOW_MeanStdSumAllsum, features4_MeanStdSum, on='user_id', how='left')

features4_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features4)
features4_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features4_mergeDataset_add = pd.merge(features4_mergeDataset,features4_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features4_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_rate', 'power_std', 'DOW_power_mean', 'DOW_power_std'], axis=1,inplace=True)

features4_Y = predicteData[(predicteData.predict_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*90, unit='D'))) & (predicteData.predict_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*91, unit='D')))]
final_features4 = pd.merge(features4_mergeDataset_add,features4_Y,on='day_of_week', how='left')

final_features4_Y = final_features4[['user_id', 'day_of_week', 'predict_date']]
final_features4_Y.rename(columns={"predict_date": "record_date"}, inplace=True)

print "predict 4st week ................."
features4_matrix = final_features4.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_Y4 = F4_xgb_model.predict(features4_matrix)
final_features4['predict_power_consumption'] = predict_Y4
final_features4_Y['power_consumption'] = predict_Y4
features4AndPredictY = pd.concat([features4, final_features4_Y], axis=0, ignore_index=True)


print "select features about 5st week............."
features5 = features4AndPredictY
features5_MeanStdSum = features5.groupby(['user_id'])['power_consumption'].agg({'power_mean':np.mean, 'power_std':np.std, 'power_sum':np.sum}).reset_index()
features5_MeanStdSum['power_rate'] = features5_MeanStdSum.power_sum / features5_MeanStdSum.power_sum.sum()
features5_DOW_MeanStdSum = features5.groupby(['user_id','day_of_week'])['power_consumption'].agg({'DOW_power_mean':np.mean, 'DOW_power_std':np.std, 'DOW_power_sum':np.sum}).reset_index()
features5_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index()
features5_DOW_MeanStdSumAllsum = features5_DOW_MeanStdSum.merge(features5_DOW_MeanStdSum.groupby('day_of_week')['DOW_power_sum'].agg({'DOW_allsum':sum}).reset_index(), on='day_of_week', how='left', copy=True)
features5_DOW_MeanStdSumAllsum['DOW_power_rate'] = features5_DOW_MeanStdSumAllsum.DOW_power_sum / features5_DOW_MeanStdSumAllsum.DOW_allsum
features5_mergeDataset = pd.merge(features5_DOW_MeanStdSumAllsum, features5_MeanStdSum, on='user_id', how='left')

features5_dayofweek_dataset = AllModelWaveletTransformByDOW.waveletTransform(features5)
features5_dayofweek_dataset.drop('level_2', axis=1,inplace=True)

features5_mergeDataset_add = pd.merge(features5_mergeDataset,features5_dayofweek_dataset,on=['user_id', 'day_of_week'], how='left')
#test_mergeDataset_add = test_mergeDataset
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_power_std', 'DOW_powaer_rate', 'power_mean', 'power_std', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_mean', 'DOW_powaer_rate', 'power_mean', 'power_rate'], axis=1,inplace=True)
#test_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum'], axis=1,inplace=True)
features5_mergeDataset_add.drop(['DOW_power_sum', 'DOW_allsum', 'power_sum', 'DOW_power_rate', 'power_rate', 'power_mean', 'power_rate', 'power_std', 'DOW_power_mean', 'DOW_power_std'], axis=1,inplace=True)

features5_Y = predicteData[(predicteData.predict_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*91, unit='D'))) & (predicteData.predict_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*91+2, unit='D')))]
final_features5 = pd.merge(features5_mergeDataset_add,features5_Y,on='day_of_week', how='left')
final_features5.dropna(inplace=True)

print "predict 5st week ................."
features5_matrix = final_features5.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
predict_Y5 = F5_xgb_model.predict(features5_matrix)
final_features5['predict_power_consumption'] = predict_Y5


print "make all features dataset ........."
features1_matrix = final_features1.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
features2_matrix = final_features2.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
features3_matrix = final_features3.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
features4_matrix = final_features4.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()
features5_matrix = final_features5.drop(['user_id', 'day_of_week', 'predict_date'], axis=1).as_matrix()


print "groupby for final_needdataset ................."
final_needdataset1 = final_features1.groupby('predict_date')['predict_power_consumption'].agg(sum).reset_index()
final_needdataset2 = final_features2.groupby('predict_date')['predict_power_consumption'].agg(sum).reset_index()
final_needdataset3 = final_features3.groupby('predict_date')['predict_power_consumption'].agg(sum).reset_index()
final_needdataset4 = final_features4.groupby('predict_date')['predict_power_consumption'].agg(sum).reset_index()
final_needdataset5 = final_features5.groupby('predict_date')['predict_power_consumption'].agg(sum).reset_index()

print "concat all final_needdataset*  (1--5) ................"
final_needdataset = pd.concat([final_needdataset1, final_needdataset2, final_needdataset3, final_needdataset4, final_needdataset5], axis=0, ignore_index=True)

print "save final_needdataset to Tianchi_power_predict_table.csv ............."
final_needdataset.to_csv(u'../result_CSV/AllModel_Tianchi_power_predict_table.csv', header=True, index=False)

def test2(x):
    return x.strftime('%Y%m%d')
final_needdataset.predict_date = final_needdataset.predict_date.apply(test2)
final_needdataset.predict_power_consumption = np.round(list(final_needdataset.predict_power_consumption)).astype('int')
final_needdataset.to_csv(u'../result_CSV/Tianchi_power_predict_table.csv', header=True, index=False)







