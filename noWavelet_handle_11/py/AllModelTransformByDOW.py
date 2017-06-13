import pandas as pd
import numpy as np

def transformByDayOfWeek(dataset):
	print "loading dataset .............." 
	def test(group):
		group.drop(['user_id', 'day_of_week','record_date'], axis=1,inplace=True)
		group.reset_index(inplace=True, drop=True)
		return group.T
	groupbydataset = dataset.groupby(['user_id', 'day_of_week']).apply(test).reset_index()
	return groupbydataset
