import csv
import pandas as pd
import random
import numpy as np
from matplotlib import pylab
import pylab as pl
from matplotlib.colors import ListedColormap
from numpy import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import *
from imblearn.combine import SMOTEENN
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
import joblib
from scipy import stats

def process_data(data):  #reduce the number of health
	sick = []
	health = []
	# tag=df.columns[0:14]
	# data=df[tag].__array__()
	for i in range(len(data)):
		if (data[i][30] == 1):
			sick.append(data[i])
		if (data[i][30] == 0):
			health.append(data[i])
	sick = np.array(sick)
	health = np.array(health)
	l = []
	sss = int(len(sick) * 0.99)
	while sss != len(l):
		x = random.randint(0, len(health))
		if x in l:
			continue
		else:
			l.append(x)
	ll = []
	for i in range(len(l)):
		ll.append(health[l[i]])
	health_sample = np.array(ll)
	dataset = []
	for ele in health_sample:
		dataset.append(ele)
	for ele in sick:
		dataset.append(ele)
	dataset = np.array(dataset)
	return dataset


def unper_sampling(train_data):  #down-sampling
	data_set = process_data(train_data)
	y_train = [x[30] for x in data_set]
	X_train = np.delete(data_set, 30, axis=1)

	return X_train, y_train
def plot_pr(auc_score, precision, recall, label=None):
	pylab.figure(num=None, figsize=(6, 5))
	pylab.xlim([0.0, 1.0])
	pylab.ylim([0.0, 1.0])
	pylab.xlabel('Recall')
	pylab.ylabel('Precision')
	pylab.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
	pylab.fill_between(recall, precision, alpha=0.5)
	pylab.grid(True, linestyle='-', color='0.75')
	pylab.plot(recall, precision, lw=1)
	pylab.show()

if __name__ == "__main__":
	print ('begin----------')
	df = pd.read_csv("data_gender_1.csv", encoding='gbk')
	X, y = df.iloc[:, 0:30].values, df.iloc[:, 30].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	train_data = np.column_stack((X_train, y_train))
	X_train, y_train = unper_sampling(train_data)

	print('------------------------Random Forest------------------------')
	param_test1 = {'min_samples_leaf': list(range(10, 20, 2)), 'n_estimators': list(range(20,110,10)),
				   'max_depth': list(range(10,60,10)),
				   'criterion': ['entropy', 'gini'],
				   }
	estimator = RandomForestClassifier(random_state=10)
	gsearch1 = GridSearchCV(estimator, param_grid=param_test1, scoring='recall', cv=5, verbose=10)

	gsearch1.fit(X_train, y_train)
	f = open("feature1.txt", "w")
	f.write(str(gsearch1.best_estimator_.feature_importances_))
	f.close()
	predicted_forest = gsearch1.predict(X_test)

	y_pred_pro_rf = gsearch1.predict_proba(X_test)[:, 1]
	y_pred_pro = gsearch1.predict_proba(X_test)

	f = open("y_pred1.txt", "w")
	f.write(str(len(y_test))+'\n')
	for i in range(len(y_test)):
		f.write(str(y_test[i])+' ')
	f.write("\n-------------------------------------------------------------------\n")
	f.write(str(len(predicted_forest))+'\n')
	for i in range(len(predicted_forest)):
		f.write(str(predicted_forest[i])+' ')
	f.write("\n-------------------------------------------------------------------\n")
	f.write(str(len(y_pred_pro_rf))+'\n')
	for i in range(len(y_pred_pro_rf)):
		f.write(str(y_pred_pro_rf[i])+' ')
	f.close()

	print('random forest: ', metrics.accuracy_score(y_test, predicted_forest))
	print(gsearch1.best_params_, gsearch1.best_score_)
	print(metrics.classification_report(y_test, predicted_forest))
	TP = 0
	FP = 0
	FN = 0
	TN = 0

	for i in range(len(y_test)):
		if ((predicted_forest[i] == 1) and (y_test[i] == 1)):
			TP += 1
		if ((predicted_forest[i] == 0) and (y_test[i] == 1)):
			FN += 1
		if ((predicted_forest[i] == 1) and (y_test[i] == 0)):
			FP += 1
		if ((predicted_forest[i] == 0) and (y_test[i] == 0)):
			TN += 1

	print('RF: TP: ', TP)
	print('RF: FP: ', FP)
	print('RF: FN: ', FN)
	print('RF: TN: ', TN)
	print('TN/(TN+FN) = 0_precision: ', TN * 1.0/(TN + FN))
	print('specificity: TN/(TN+FP) = 0_recall: ', TN * 1.0 / (TN + FP))
	print('TP/(TP+FP) = 1_precision: ', TP * 1.0 / (TP + FP))
	print('sensitivity: TP/(TP+FN) = 1_recall: ', TP * 1.0 / (TP + FN))
	print('yuden = sensitivity + specificity - 1', TP * 1.0 / (TP + FN) + TN * 1.0 / (TN + FP) - 1)
	print('confusion_matrix: ', metrics.confusion_matrix(y_test, predicted_forest))
	print('------------------------Random Forest------------------------')

	precision, recall, thresholds = precision_recall_curve(y_test, y_pred_pro_rf)
	plot_pr(0.5, precision, recall, "pos")
	print(y_pred_pro)

	KS, p = stats.ks_2samp(y_test, predicted_forest)
	print('KS = ', KS, 'p = ', p)

	fpr, tpr, threshold = roc_curve(y_test, y_pred_pro[:, 0], pos_label=0)
#    print (fpr)
#    print (tpr)
	C = np.zeros(len(tpr))
	for i in range(len(tpr)):
		C[i] = tpr[i] - fpr[i]
#    print (C)
	cutoff = np.max(C)
	print ('cutoff = ', cutoff)
	roc_auc = auc(fpr, tpr)
	print ('auc = ', roc_auc)

	plt.figure(figsize=(10,10))
	plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc="lower right")
	plt.show()
	print('end--------------------')
