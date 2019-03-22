import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import csv
import time
import math
from sklearn import preprocessing, cross_validation, neighbors
import compute_measure
import pandas as pd
from numpy import genfromtxt, savetxt
import statistics
from sklearn.cluster import KMeans

df = pd.read_csv("data.csv")

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#split 10% of the dataset to test the performance of each classifiers
A_train, A_test, b_train, b_test = cross_validation.train_test_split(X_train,y_train,test_size=0.2)

names = [ "k-NN",
		 "DecisionTree",
		 "RandomForest",
		 "AdaBoost",
		 "BaggingRegressor",
		 "Gradient Boosting",
		 "MLP",
		 "Bayesian Ridge"]


regressors = [
	KNeighborsRegressor(3),
	DecisionTreeRegressor(max_depth=5),
	RandomForestRegressor(max_depth=5, n_estimators=10, max_features=1),
	AdaBoostRegressor(),
	BaggingRegressor(),
	GradientBoostingRegressor(),
	MLPRegressor(alpha=1e-5, hidden_layer_sizes=(100, 1),  max_iter=5000),
	linear_model.BayesianRidge()]


def get_MSE(Error):
	mse=np.sum(np.power(Error,2))
	mse=mse/len(Error)
	return mse

copy_names = list(names)
MSE = []
for name, reg in zip(names, regressors):
	reg.fit(A_train, b_train)
	Error_reg   = [None] * len(A_train)
	predictedIV = reg.predict(A_test)
	Error_reg   = abs(b_test - predictedIV)
	print('\n\nThe Model '+ name + ' Peformance Summary as follows:')
	print('The MSE is           {:20.16f}'.format(get_MSE(Error_reg)))
	print('The mean error is    {:20.16f}'.format(np.mean(Error_reg)))
	print('The maximum error is {:20.16f}'.format(max(Error_reg)))
	print('The minimum error is {:20.16f}'.format(min(Error_reg)))
	MSE.append(get_MSE(Error_reg))

new_regressors = sorted(zip(MSE, copy_names, regressors))[:3]
##store value.

print("\n\nTop Three Regressors:")

for MSE, name, reg in new_regressors:
	print(name)
	print('The MSE is           {:20.16f}'.format(MSE))


print("\n\n\n")
print("Consensus Learning Method:")


prediction_results = []

new_MSE = []
new_name_list = []
new_regressors_list = []
for MSE, name, reg in new_regressors:
	reg.fit(X_train, y_train)
	Error_reg   = [None] * len(X_train)
	predictedIV = reg.predict(X_test)
	prediction_results.append(predictedIV)
	Error_reg   = abs(y_test - predictedIV)
	print('\n\nThe Model '+ name + ' Peformance Summary as follows:')
	print('The MSE is           {:20.16f}'.format(get_MSE(Error_reg)))
	print('The mean error is    {:20.16f}'.format(np.mean(Error_reg)))
	print('The maximum error is {:20.16f}'.format(max(Error_reg)))
	print('The minimum error is {:20.16f}'.format(min(Error_reg)))
	new_MSE.append(get_MSE(Error_reg))
	new_name_list.append(name)
print(prediction_results)

sorted_regressors = sorted(zip(new_MSE, new_name_list, prediction_results))
print('\n\nSorted:')
sorted_new_MSE = []
for MSE, name, pred in sorted_regressors:
	print(name)
	print(MSE)
	sorted_new_MSE.append(MSE)

# weighting algorithm #1
weights_1 = (1/sorted_new_MSE[0])/(1/np.sum(sorted_new_MSE))
weights_2 = (1/sorted_new_MSE[1])/(1/np.sum(sorted_new_MSE))
weights_3 = (1/sorted_new_MSE[2])/(1/np.sum(sorted_new_MSE))

# weighting algorithm #2
weights_4 = (1/math.pow(sorted_new_MSE[0], 2))/(1/math.pow(np.sum(sorted_new_MSE), 2))
weights_5 = (1/math.pow(sorted_new_MSE[1], 2))/(1/math.pow(np.sum(sorted_new_MSE), 2))
weights_6 = (1/math.pow(sorted_new_MSE[2], 2))/(1/math.pow(np.sum(sorted_new_MSE), 2))

# weighting algorithm #2
weights_7 = (1/math.log(math.pow(sorted_new_MSE[0], 2)))/(1/math.log(math.pow(np.sum(sorted_new_MSE), 2)))
weights_8 = (1/math.log(math.pow(sorted_new_MSE[1], 2)))/(1/math.log(math.pow(np.sum(sorted_new_MSE), 2)))
weights_9 = (1/math.log(math.pow(sorted_new_MSE[2], 2)))/(1/math.log(math.pow(np.sum(sorted_new_MSE), 2)))

#the first prediction for using the first grounp of weights:
print("\n\n\n")
merged_1 = np.zeros_like(prediction_results[0])
merged_1.fill(0)

prediction_test1=[]
prediction_test2=[]
prediction_test3=[]
# for i in range(merged_1.size):
# 	prediction_sum = prediction_results[0][i]*weights_1 + prediction_results[1][i]*weights_2 + prediction_results[2][i]*weights_3
# 	prediction_test1.append(prediction_sum)
# 	if prediction_sum >= 0.9:
# 		merged_1[i] = 1
# 	if prediction_sum <= -0.9:
# 		merged_1[i] = -1
# 	if prediction_sum < 0.9 and prediction_sum > -0.9:
# 		merged_1[i] = 0
# print(prediction_test1)
# firstmode=statistics.mode(prediction_test1)
# print(firstmode)

print("\n\n\n")
prediction_sum = np.zeros_like(prediction_results[0])
prediction_sum.fill(0)

for i in range(y_test.size):
     prediction_sum[i] = prediction_results[0][i] * weights_1 + prediction_results[1][i] * weights_2 + \
                        prediction_results[2][i] * weights_3
print(prediction_sum)
kmeans = KMeans(n_clusters=2, random_state=0).fit(prediction_sum.reshape(-1, 1))
kmeans.cluster_centers_()
firstmode=statistics.mode(prediction_sum)
print(firstmode)

# the second prediction for using the second grounp of weights:
# merged_2 = np.zeros_like(prediction_results[0])
# merged_2.fill(0)
#
#
# for i in range(merged_2.size):
# 	prediction_sum = prediction_results[0][i]*weights_4 + prediction_results[1][i]*weights_5 + prediction_results[2][i]*weights_6
# 	prediction_test2.append(prediction_sum)
# 	if prediction_sum >= 0.9:
# 		merged_2[i] = 1
# 	if prediction_sum <= -0.9:
# 		merged_2[i] = -1
# 	if prediction_sum < 0.9 and prediction_sum > -0.9:
# 		merged_2[i] = 0
#
# secondmode = statistics.mode(prediction_test2)
# print(secondmode)
# # the third prediction for using the third grounp of weights:
# merged_3 = np.zeros_like(prediction_results[0])
# merged_3.fill(0)
#
#
# for i in range(merged_3.size):
# 	prediction_sum = prediction_results[0][i]*weights_7 + prediction_results[1][i]*weights_8 + prediction_results[2][i]*weights_9
# 	prediction_test3.append(prediction_sum)
# 	if prediction_sum >= 0.9:
# 		merged_3[i] = 1
# 	if prediction_sum <= -0.9:
# 		merged_3[i] = -1
# 	if prediction_sum < 0.9 and prediction_sum > -0.9:
# 		merged_3[i] = 0
#
# thirdmode = statistics.mode(prediction_test3)
# print(thirdmode)
#
# counter_1 = 0
# counter_positive_1 = 0
# for i in range(merged_1.size):
#     counter_1 = counter_1 + 1
#     if merged_1[i] != 0:
#         if merged_1[i] == y_test[i]:
#             counter_positive_1 = counter_positive_1 + 1
#
#
# counter_2 = 0
# counter_positive_2 = 0
# for i in range(merged_1.size):
#     counter_2 = counter_2 + 1
#     if merged_2[i] != 0:
#         if merged_2[i] == y_test[i]:
#             counter_positive_2 = counter_positive_2 + 1
#
#
# counter_3 = 0
# counter_positive_3 = 0
# for i in range(merged_1.size):
#     counter_3 = counter_3 + 1
#     if merged_3[i] != 0:
#         if merged_3[i] == y_test[i]:
#             counter_positive_3 = counter_positive_3 + 1