import pandas as pd
import numpy as np
import pywt

def waveletTransform(dataset):
	print "loading dataset .............." 
	def test(group):
		group.reset_index(inplace=True, drop=True)
		waveletGroup = pywt.wavedec(group.power_consumption, 'db2', 'zero')
		wavelet_list = []
		wavelet_list.extend(waveletGroup[0])
		wavelet_list.extend(waveletGroup[1])
		wavelet_list.extend(waveletGroup[2])
		newgroup = pd.DataFrame([wavelet_list], columns=range(len(wavelet_list)))
		return newgroup
	groupbydataset = dataset.groupby(['user_id', 'day_of_week']).apply(test).reset_index()
	print groupbydataset.shape
	print len(groupbydataset.dropna())
	return groupbydataset
