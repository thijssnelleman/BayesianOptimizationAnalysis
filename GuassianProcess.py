import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
import time

class GaussianProcess():
	def __init__(self, filename, nr_calls, group_runs=10):
		dataset = pd.read_csv(filename)
		dataset.shape
		dataset.head()
		self.X = dataset.drop('y', axis=1)
		self.y = dataset['y']
		self.nr_calls = nr_calls
		self.group_runs=group_runs
		
		self.SPACE = [
			Categorical(['mse', 'friedman_mse', 'mae'], name='criterion'),
			Categorical(['best','random'], name='splitter'),
			Integer(1, dataset.size, name='max_depth'),
			Integer(2, dataset.size, name='min_samples_split'),
			Integer(1, dataset.size, name='min_samples_leaf'),
			Real(0.0, 0.5, name='min_weight_fraction_leaf'),
			Categorical(['auto','sqrt','log2',None], name='max_features'),
			Integer(2, dataset.size, name='max_leaf_nodes'),
			Real(0.0, 1.0, name='min_impurity_decrease'),
			Real(0.0, 1.0, name='ccp_alpha')]
			
	
	def f(self, params):
		X_train, X_test, y_train, y_test = train_test_split(
											self.X, self.y, test_size=0.2)
		classifier = DecisionTreeRegressor(
						**{dim.name: val for dim, val in
							zip(self.SPACE, params) if dim.name != 'dummy'})
		classifier.fit(X_train, y_train)
		pred = classifier.predict(X_test)
		s = abs(y_test - pred).sum()
		return s
	
	
	def run(self):
		clf=gp_minimize(self.f,
						self.SPACE,
						acq_func="EI",
						n_calls=self.nr_calls,
						n_initial_points=15,
						noise=0.01,
						random_state=None,
						n_jobs=-1)
		res = []
		minFx = 100000000
		for i in clf.func_vals:
			if i < minFx:
				minFx = i
			res.append(minFx)
		return res
		
	
	def run_groups(self):
		start = time.time()
		res = []
		for i in range (self.group_runs):
			res.append(self.run())
		end = time.time()
		print("finished GP runs in", end-start, "seconds")
		return res
