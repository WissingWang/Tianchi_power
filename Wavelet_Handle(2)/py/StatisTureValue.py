import pandas as pd
import numpy as np
import pywt
from datetime import datetime
from dateutil.parser import parse
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

dataset = handledataset[(handledataset.record_date>=pd.to_datetime('2015-01-01')+pd.to_timedelta(7*82, unit='D')) & (handledataset.record_date<(pd.to_datetime('2015-01-01')+pd.to_timedelta(7*86+2, unit='D')))]

final_dataset = dataset.groupby('record_date')['power_consumption'].agg({"predict_power_consumption":sum}).reset_index()
final_dataset.rename(columns={'record_date': 'predict_date'}, inplace=True)
def test1(x):
    return x.strftime('%Y%m%d')
final_dataset.predict_date = final_dataset.predict_date.apply(test1)
final_dataset.predict_power_consumption = np.round(list(final_dataset.predict_power_consumption)).astype('int')
final_dataset.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle/statisTrueValue.csv', header=True, index=False)


predict_dataset = pd.read_csv(u'/home/haven/Tianchi_power/Wavelet_Handle/AllModel_Tianchi_power_predict_table_test.csv')
def test2(x):
    return parse(x).strftime('%Y%m%d')
predict_dataset.predict_date = predict_dataset.predict_date.apply(test2)
predict_dataset.predict_power_consumption = np.round(list(predict_dataset.predict_power_consumption)).astype('int')
predict_dataset.to_csv(u'/home/haven/Tianchi_power/Wavelet_Handle/Tianchi_power_predict_table_test.csv', header=True, index=False)

print "MSE = ",metrics.mean_squared_error(final_dataset.predict_power_consumption, predict_dataset.predict_power_consumption)

print "corrcoef = ",np.corrcoef(predict_dataset.predict_power_consumption, final_dataset.predict_power_consumption)
