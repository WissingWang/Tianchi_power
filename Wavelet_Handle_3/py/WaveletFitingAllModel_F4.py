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
train1 = pd.read_csv(u'/home/haven/Tianchi_power/Wavelet_Handle_3/F3_Result/train1AndPredictY.csv')
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
train1_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*81, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*82, unit='D')))]
final_train1 = pd.merge(train1_mergeDataset_add,train1_Y,on=['user_id', 'day_of_week'], how='left')



print "select train2 dataset ............."
train2 = pd.read_csv(u'/home/haven/Tianchi_power/Wavelet_Handle_3/F3_Result/train2AndPredictY.csv')
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
train2_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*82, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*83, unit='D')))]
final_train2 = pd.merge(train2_mergeDataset_add,train2_Y,on=['user_id', 'day_of_week'], how='left')




print "select train3 dataset ............."
train3 = pd.read_csv(u'/home/haven/Tianchi_power/Wavelet_Handle_3/F3_Result/train3AndPredictY.csv')
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
train3_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*83, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*84, unit='D')))]
final_train3 = pd.merge(train3_mergeDataset_add,train3_Y,on=['user_id', 'day_of_week'], how='left')




print "select train4 dataset ............."
train4 = pd.read_csv(u'/home/haven/Tianchi_power/Wavelet_Handle_3/F3_Result/train4AndPredictY.csv')
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

train4_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*84, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*85, unit='D')))]

final_train4 = pd.merge(train4_mergeDataset_add,train4_Y,on=['user_id', 'day_of_week'], how='left')


print "select test dataset ............."
test = pd.read_csv(u'/home/haven/Tianchi_power/Wavelet_Handle_3/F3_Result/testAndPredictY.csv')
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

test_Y = handledataset[(handledataset.record_date>=(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*85, unit='D'))) & (handledataset.record_date<(pd.to_datetime('2015-01-01') + pd.to_timedelta(7*86, unit='D')))]

final_test = pd.merge(test_mergeDataset_add,test_Y,on=['user_id', 'day_of_week'], how='left')



print "make all train dataset ........."
#train = pd.concat([final_train1, final_train2, final_train3, final_train4], axis=0, ignore_index=True)
train1_matrix = final_train1.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train2_matrix = final_train2.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train3_matrix = final_train3.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
train4_matrix = final_train4.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
#final_train_matrix = train.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
final_train_matrix = np.row_stack((train1_matrix, train2_matrix, train3_matrix, train4_matrix))
train_X = final_train_matrix[:,:-1]
train_Y = final_train_matrix[:,-1]

print "make test datset"
final_test_matrix = final_test.drop(['user_id', 'day_of_week', 'record_date'], axis=1).as_matrix()
test_matrix_X = final_test_matrix[:,:-1]
test_matrix_Y = final_test_matrix[:,-1]


print("hyper-parameter optimization...................")
xgb_model = xgb.XGBRegressor()
params = {'max_depth':[2,3,4,5,6], 'learning_rate':[0.05,0.1,0.15], 'n_estimators':[50,100,150,200], 'max_delta_step':[1] ,
          'objective':['reg:linear', 'reg:gamma','reg:tweedie',]}
# , 'colsample_bytree':[1], 'colsample_bylevel':[1], 'reg_alpha':[0], 'reg_lambda':[1], 'scale_pos_weight':[1], 'base_score':[0.5], 'seed':[0], 'missing':[None],'nthread':[-1], 'gamma':[0], 'min_child_weight':[1], , 'subsample':[0.5,0.8,1]
gridsearchcvRegression = GridSearchCV(xgb_model, params, iid=True,scoring=None, n_jobs=1, refit=True, verbose=2, return_train_score=True)

print "optimization fitting ..............."
gridsearchcvRegression.fit(train_X,train_Y)

print "/n"
print "Best Score : ",gridsearchcvRegression.best_score_
print "Best Params : ",gridsearchcvRegression.best_params_

print "predict fiting............."
xgb_model = xgb.XGBRegressor(n_estimators=gridsearchcvRegression.best_params_['n_estimators'],max_depth=gridsearchcvRegression.best_params_['max_depth'],objective=gridsearchcvRegression.best_params_['objective'], max_delta_step=1, learning_rate=gridsearchcvRegression.best_params_['learning_rate'],silent=False)
xgb_model.fit(train_X, train_Y)

input_flag = raw_input("Do you want to save the model by train? Input[y/n]:")
if input_flag == 'y':
	joblib.dump(xgb_model, "../model/F4_xgb_model.m")
print "predict ............"
predict_Y = xgb_model.predict(test_matrix_X)
predict_Y = np.round(predict_Y).astype(int)

print "MSE = ",metrics.mean_squared_error(test_matrix_Y, predict_Y)

summ =0
for i in range(len(test_matrix_Y)):
    if (predict_Y[i]+test_matrix_Y[i]) == 0:
        continue
    summ+=abs((predict_Y[i]-test_matrix_Y[i])/(predict_Y[i]+test_matrix_Y[i]))
meansum = summ/len(test_matrix_Y)
print "MeanSum = ",meansum

print "corrcoef = ",np.corrcoef(predict_Y, test_matrix_Y)

print "create next datasets ..........."
train1_matrix_X = train1_matrix[:,:-1]
train1_predict_Y = xgb_model.predict(train1_matrix_X)
train1_Y['power_consumption'] = train1_predict_Y
train1AndPredictY = pd.concat([train1, train1_Y], axis=0, ignore_index=True)
train1AndPredictY.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle_3/F4_Result/train1AndPredictY.csv', header=True, index=False)

train2_matrix_X = train2_matrix[:,:-1]
train2_predict_Y = xgb_model.predict(train2_matrix_X)
train2_Y['power_consumption'] = train2_predict_Y
train2AndPredictY = pd.concat([train2, train2_Y], axis=0, ignore_index=True)
train2AndPredictY.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle_3/F4_Result/train2AndPredictY.csv', header=True, index=False)

train3_matrix_X = train3_matrix[:,:-1]
train3_predict_Y = xgb_model.predict(train3_matrix_X)
train3_Y['power_consumption'] = train3_predict_Y
train3AndPredictY = pd.concat([train3, train3_Y], axis=0, ignore_index=True)
train3AndPredictY.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle_3/F4_Result/train3AndPredictY.csv', header=True, index=False)

train4_matrix_X = train4_matrix[:,:-1]
train4_predict_Y = xgb_model.predict(train4_matrix_X)
train4_Y['power_consumption'] = train4_predict_Y
train4AndPredictY = pd.concat([train4, train4_Y], axis=0, ignore_index=True)
train4AndPredictY.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle_3/F4_Result/train4AndPredictY.csv', header=True, index=False)

test_Y['power_consumption'] = predict_Y
testAndPredictY = pd.concat([test, test_Y], axis=0, ignore_index=True)
testAndPredictY.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle_3/F4_Result/testAndPredictY.csv', header=True, index=False)


