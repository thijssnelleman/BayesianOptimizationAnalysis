import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from skopt.space import Integer, Categorical, Real
import time

class RandomSearch():
	def __init__(self, filename, nr_calls, group_runs=10):
		dataset = pd.read_csv(filename)
		dataset.shape
		dataset.head()
		self.X = dataset.drop('y', axis=1)
		self.y = dataset['y']
		self.nr_calls = nr_calls
		self.group_runs=group_runs
		
		self.parameters = {
			'criterion':['mse', 'friedman_mse', 'mae'],
			'splitter':['best','random'],
			'max_features':['auto','sqrt','log2',None],
			'max_depth':np.linspace(1, dataset.size,
									#num=50, 
									dtype=int),
			'min_samples_split':np.linspace(2, dataset.size,
											#num=50,
											dtype=int),
			'min_samples_leaf':np.linspace(1, dataset.size,
											#num=50,
											dtype=int),
			'min_weight_fraction_leaf': np.linspace(0.0,0.5,num=50),
			'max_leaf_nodes':np.linspace(2,dataset.size,
											#num=50,
											dtype=int),
			'min_impurity_decrease':np.linspace(0.0,1.0),#,num=50),
			'ccp_alpha':np.linspace(0.0,1.0)}#,num=50)}


	def run(self):
		res = []
		minFx = 100000000
		for i in range(self.nr_calls):
			X_train, X_test, y_train, y_test = train_test_split(
											self.X, self.y, test_size=0.2)
			clf = RandomizedSearchCV(DecisionTreeRegressor(),
									self.parameters,
									n_iter=1,
									scoring='r2')
			clf.fit(X_train, y_train)
			pred = clf.predict(X_test)
			s = abs(y_test - pred).sum()
			if s < minFx:
				minFx = s
			res.append(minFx)
			del clf
		return res
	
		
	def run_groups(self):
		start = time.time()
		res = []
		for i in range (self.group_runs):
			res.append(self.run())
		end = time.time()
		print("finished RS runs in ", end-start, "seconds")
		return res
