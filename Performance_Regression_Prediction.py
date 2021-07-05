### pip install dtreeviz
### pip install shapash
### pip install imbalanced-learn

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import imblearn

import io

from sklearn import tree

from dtreeviz.trees import dtreeviz

import graphviz

from shapash.explainer.smart_explainer import SmartExplainer

from scipy.stats import pearsonr

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import classification_report, f1_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

# from sklearn.naive_bayes import GaussianNB

# from sklearn.linear_model import LogisticRegression, SGDClassifier

# from sklearn.neural_network import MLPClassifier





def feature_extraction(file):
	"""This function reads the data from the csv file and extract the neccessary columns from it.
		Also, it will chec the null value in this dataset.

		Return
		------
		desired features dataset: dataframe
	"""

	dataset = pd.read_csv(file)

	#Check for null values

	#Deal with the null values with the median
	dataset = dataset.fillna(dataset.median())

	#After cleaning the data you have to shuffle the dataset
	dataset = dataset.sample(frac = 1)

	#Dataset - eliminate the unnecessary features/columns(Commit A, B, Benchmark, Hit/Dismiis are not training features.)
	df2_y = dataset['Hit/Dismiss']
	df2 = dataset.drop(labels=['Commit A','Commit B','Benchmark','Hit/Dismiss'],axis=1)

	return df2, df2_y


def train_test_validation(df2, df2_y):
	"""split the dataset into test, train, validation
		train:0.8*0.75, test:0.2, validation: train(0.8)*0.25
		return
		------
		X_train: Training dataset contain 60% of dataset, dataframe.
		y_train: Training label contains 60% of dataset, dataframe.
		X_val: Validation dataset contains 20% of dataset, dataframe.
		y_val: Validation label contains 20% of dataset, dataframe.
		X_test: Testing dataset contain 20% of dataset, dataframe
		y_test: Testing label contains 20% of dataset, dataframe
	"""

	# test:0.2, train:0.8 has the best result right now.
	X_train,X_test,y_train,y_test = train_test_split(df2, df2_y, test_size = 0.2, random_state = 1)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 1)

	return X_train, y_train, X_val, y_val, X_test, y_test



def feature_selection_shapash(df2):
	"""#Shapash - Visual aid for feature selection
		Description and Context - https://www.kdnuggets.com/2021/04/shapash-machine-learning-models-understandable.html
		Link - https://pypi.org/project/shapash/
	"""

	xpl = SmartExplainer()

	xpl.compile(
	    x=df2,
	    model=clf_RFC,
	)

	xpl.plot.features_importance()


def feature_selection_pearsonr(df2):
	"""Scipy Pearson Correlation Coefficient """

	#Drop some useless features

	dataset.drop('Top Chg by Instr >= X%', axis = 1, inplace = True)
	df2.drop('Top Chg by Instr >= X%',axis = 1,inplace = True)

	#correlation of all the features in dataset.
	corr = df2.corr()

	# Build a map to know the correlation value over 0.7 or smaller than -0.7.
	corr2 = corr[(abs(corr)> 0.7)]

	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(11, 9))

	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(230, 20, as_cmap=True)

	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr2, cmap='viridis', center=0,
	            square=True, linewidths=.5, cbar_kws={"shrink": .5})

def grid_search_knn(classifier, X_train, y_train, n_neighbors = [3, 5, 7, 9], weights = ['uniform','distance'], algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute'], p = [1,2]):
	"""Use the grid search to do the hyperparameter tuning on KNN.
		Parameter:	n_neighbors: [3, 5, 7, 9}
					weights: [uniform, distance]
					algorithm: [auto, ball_tree, kd_tree, brute]
					p: [1,2]

		return
		------
		The best parameter value we could get for now.
	"""
	param_grid_KNN = { 'n_neighbors' : n_neighbors ,
	              'weights' : weights ,
	              'algorithm' : algorithm ,
	              'p' : p
	              		}

	gridSearchCV_KNN = GridSearchCV(classifier, param_grid = param_grid_KNN, scoring = 'f1')

	gridSearchCV_KNN.fit(X_train, y_train)

	return gridSearchCV_KNN.best_estimator_.get_params()

def grid_search_rfc(classifier, X_train, y_train, bootstrap = [True, False], max_depth = [10, 50], max_features = ['auto', 'sqrt'], min_samples_leaf = [1, 2], min_samples_split = [2, 5], n_estimators = [400, 800, 1200]):
	"""Use the grid search to do the hyperparameter tuning on RFC
		Parameter:
		bootstrap: [True, False]
		max_depth: [10, 50],
		max_features: ['auto', 'sqrt'],
		min_samples_leaf: [1, 2],
		min_samples_split: [2, 5],
		n_estimators: [400, 800, 1200]
	"""

	param_grid_RFC = {	'bootstrap': bootstrap,
					'max_depth': max_depth,
 					'max_features': max_features,
 					'min_samples_leaf': min_samples_leaf,
 					'min_samples_split': min_samples_split,
 					'n_estimators': n_estimators
 						}

	gridSearchCV_RFC = GridSearchCV(classifier, param_grid = param_grid_RFC, scoring = 'f1')

	gridSearchCV_RFC.fit(X_train, y_train)

	return gridSearchCV_RFC.best_estimator_.get_params()


if __name__ == "__main__":

	#Get the expected features from the dataset
	df, df_y = feature_extraction('DataSetCuratedExtended2020.csv')

	#Do the feature selection through shapash
	# feature_selection_shapash(df)

	#DataFrame with selected features via shapash
	df_final = df[["SumEssential","CountLine","CountLineCodeDecl","CountLineBlank","AltCountLineCode","AvgLineComment",'AvgEssential','AvgLine',"New Func >= X","AltAvgLineComment","Top Reached Chg Len >= X%","CountLineCodeExe",'CountStmt']]

	#split the df_final into training, validation, testing
	X_train, y_train, X_val, y_val, X_test, y_test = train_test_validation(df_final, df_y)

	#imbalance issue--> implement oversample and undersample. Put the default value in 0.5.
	oversample = RandomOverSampler(sampling_strategy=0.5)

	undersample = RandomUnderSampler(sampling_strategy=0.5)


	#KNNClassifier - oversampling

	clf_KNN = KNeighborsClassifier(n_neighbors=4)

	X_train_over,y_train_over = oversample.fit_resample(X_train,y_train)
	clf_KNN.fit(X_train_over,y_train_over)
	predictions = clf_KNN.predict(X_val)
	print("Result: KNN classifier")
	print(classification_report(y_val,predictions))


	#Random Forest Classifier - oversampling
	clf_RFC = RandomForestClassifier()

	X_train_over,y_train_over = oversample.fit_resample(X_train,y_train)
	clf_RFC.fit(X_train_over,y_train_over)
	predictions = clf_RFC.predict(X_val)
	print("Result: Random Forest classifier")
	print(classification_report(y_val,predictions))

	#cross validation
	print("validation score - KNN:")
	print(cross_val_score(clf_KNN, X_val, y_val,cv=10,scoring='f1'))

	print("validation score - RFC:")
	print(cross_val_score(clf_RFC, X_val, y_val,cv=10,scoring='f1'))


	# #Hyper parameter tuning(GridSearch)-KNN
	grid_result_KNN = grid_search_knn(clf_KNN, X_train_over, y_train_over)


	#run the model again after grid search
	clf_KNN = KNeighborsClassifier(algorithm = grid_result_KNN['algorithm'], n_neighbors = grid_result_KNN['n_neighbors'], weights = grid_result_KNN['weights'], p = grid_result_KNN['p'])

	X_train_over,y_train_over = oversample.fit_resample(X_train,y_train)
	clf_KNN.fit(X_train_over,y_train_over)
	predictions = clf_KNN.predict(X_test)
	print("Result after parameter tuning: KNN classifier")
	print(classification_report(y_test,predictions))
