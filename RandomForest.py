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
import cmath
from sklearn.metrics import roc_auc_score

def process_data(data):  #reduce the number of health
	sick = []
	health = []
	# tag=df.columns[0:14]
	# data=df[tag].__array__()
	for i in range(len(data)):
		if (data[i][21] == 1):
			sick.append(data[i])
		if (data[i][21] == 0):
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
	y_train = [x[21] for x in data_set]
	X_train = np.delete(data_set, 21, axis=1)

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
	df = pd.read_csv("data_gender_0.csv", encoding='gbk')
	X, y = df.iloc[:, 0:21].values, df.iloc[:, 21].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	train_data = np.column_stack((X_train, y_train))
	X_train, y_train = unper_sampling(train_data)

	print('------------------------Random Forest------------------------')
	param_test1 = {'min_samples_leaf': list(range(10, 20, 2)), 'n_estimators': list(range(20, 110, 10)),
				   'max_depth': list(range(10, 60, 10)),
				   'criterion': ['entropy', 'gini'],
				   }
	estimator = RandomForestClassifier(random_state=10)
	gsearch1 = GridSearchCV(estimator, param_grid=param_test1, scoring='recall', cv=5, verbose=10)

	gsearch1.fit(X_train, y_train)
	f = open("feature.txt", "w")
	f.write(str(gsearch1.best_estimator_.feature_importances_))
	f.close()
	predicted_forest = gsearch1.predict(X_test)

	y_pred_pro_rf = gsearch1.predict_proba(X_test)[:, 1]
	y_pred_pro = gsearch1.predict_proba(X_test)


	#导出训练集为A.xlsx，导出测试集+预测值+预测概率为B.xlsx
	A = np.column_stack((X_train, y_train))
	B = np.column_stack((X_test, y_test))
	B = np.column_stack((B, predicted_forest))
	B = np.column_stack((B, y_pred_pro_rf))
	data = pd.DataFrame(A)
	writer = pd.ExcelWriter('A.xlsx')  # 写入Excel文件
	data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
	writer.save()
	writer.close()
	data = pd.DataFrame(B)
	writer = pd.ExcelWriter('B.xlsx')  # 写入Excel文件
	data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
	writer.save()
	writer.close()

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

	#灵敏度最大时、 max{TPR*TNR} 以及 max{TPR+TNR}三种情况分别为cutoff1，cutoff2，cutoff3；对应tpr1、tnr1；tpr2、tnr2；tpr3、tnr3。
	fpr, tpr, threshold = roc_curve(y_test, y_pred_pro[:, 0], pos_label=0)
#    print (fpr)
#    print (tpr)
	C = np.zeros(len(tpr))
	l = 0
	for i in range(len(tpr)):
		C[i] = tpr[i]
		if tpr[i] > l:
			l = i
#    print (C)
	cutoff1 = np.max(C)
	print('cutoff1 = ', cutoff1)
	print('tpr1 = ', tpr[l])
	print('tnr2 = ', 1 - fpr[l])

	D = np.zeros(len(tpr))
	j = 0
	for i in range(len(tpr)):
		D[i] = (tpr[i] * (1-fpr[i]))
		if D[i] > j:
			j = i
	cutoff2 = np.max(D)
	print('cutoff2 = ', cutoff2)
	print('tpr2 = ', tpr[j])
	print('tnr2 = ', 1-fpr[j])

	E = np.zeros(len(tpr))
	k = 0
	for i in range(len(tpr)):
		E[i] = tpr[i] + (1-fpr[i])
		if E[i] > k:
			k = i
	cutoff3 = np.max(E)
	print('cutoff3 = ', cutoff3)
	print('tpr3 = ', tpr[k])
	print('tnr3 = ', 1-fpr[k])

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


	#置信区间0.95;
	print("Original ROC area: {:0.3f}".format(roc_auc_score(y_test, y_pred_pro_rf)))
	n_bootstraps = 1000
	rng_seed = 42  # control reproducibility
	bootstrapped_scores = []
	rng = np.random.RandomState(rng_seed)
	for i in range(n_bootstraps):
		# bootstrap by sampling with replacement on the prediction indices
		indices = rng.random_integers(0, len(y_pred_pro_rf) - 1, len(y_pred_pro_rf))
		if len(np.unique(y_test[indices])) < 2:
			# We need at least one positive and one negative sample for ROC AUC
			# to be defined: reject the sample
			continue

		score = roc_auc_score(y_test[indices], y_pred_pro_rf[indices])
		bootstrapped_scores.append(score)
		print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
	plt.hist(bootstrapped_scores, bins=50)
	plt.title('Histogram of the bootstrapped ROC AUC scores')
	plt.show()
	sorted_scores = np.array(bootstrapped_scores)
	sorted_scores.sort()
	# Computing the lower and upper bound of the 90% confidence interval
	# You can change the bounds percentiles to 0.025 and 0.975 to get
	# a 95% confidence interval instead.
	confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
	confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
	print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
		confidence_lower, confidence_upper))
