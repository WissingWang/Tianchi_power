import pandas as pd
import numpy as np

def transformByDayOfWeek():
	print "loading dataset .............." 
	handledataset = pd.read_csv(u'/home/haven/Tianchi_power/Tianchi_power__boxhandle15_DayOfWeek.csv')
	def test(group):
		group.drop(['user_id', 'day_of_week','record_date'], axis=1,inplace=True)
		group.reset_index(inplace=True, drop=True)
		return group.T
	groupbydataset = handledataset.groupby(['user_id', 'day_of_week']).apply(test).reset_index()
	return groupbydataset
