import numpy as np
import pandas as pd
from sklearn import tree
import helpers.datasets.adult as adult
from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import time

import helpers.datasets.statlog as statlog

adult_data = adult.load('learning', encode_features=True)
adult_data = adult.to_numpy_array(adult_data, remove_missing_values=True)

# Separating to target and features
adult_targets = adult_data[:, -1]
adult_data = adult_data[:, 0:-1]

# n = np.shape(adult_data)[0]
# print(n)
#
# subset_over_50k = len(np.where(adult_data[:, -1] == 1)[0]) / n
# subset_less_50k = len(np.where(adult_data[:, -1] == 0)[0]) / n
#
# print(subset_less_50k, subset_over_50k)
#
# # adult.print_feature_subsets_proportions(adult_data, "Race")
# # adult.print_feature_subsets_proportions(adult_data, "Sex")
# # adult.print_feature_subsets_proportions(adult_data, "Country")
# # adult.print_feature_subsets_proportions(adult_data, "Age")
#
# adult_data = adult.load('testing', encode_features=True)
# adult_data = adult.to_numpy_array(adult_data, remove_missing_values=True)
#
# n = np.shape(adult_data)[0]
# print(n)
#
# subset_over_50k = len(np.where(adult_data[:, -1] == 1)[0]) / n
# subset_less_50k = len(np.where(adult_data[:, -1] == 0)[0]) / n
#
# print(subset_less_50k, subset_over_50k)

# adult.print_feature_subsets_proportions(adult_data, "Race")
# adult.print_feature_subsets_proportions(adult_data, "Sex")
# adult.print_feature_subsets_proportions(adult_data, "Country")
# adult.print_feature_subsets_proportions(adult_data, "Age")

# models = [tree.DecisionTreeClassifier(max_depth=3, criterion="entropy"),
#           RandomForestClassifier(n_estimators=50, max_features=None, max_depth=14),
#           make_pipeline(StandardScaler(), LogisticRegression(C=6.6, penalty='l1', tol=0.01)),
#           make_pipeline(StandardScaler(), GaussianNB()),
#           make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
#           make_pipeline(StandardScaler(), svm.SVC(kernel="rbf", C=100, gamma=0.001)),
#           make_pipeline(StandardScaler(), MLPClassifier(activation='tanh', epsilon=0.001, hidden_layer_sizes=(10,),
#                                                         solver='lbfgs', tol=1e-06))]
#
# model_names = ["DT", "RF", "Logistic Regression", "GuassianNB", "KNN", "SVC", "MLP"]
#
# for clf_index in range(len(models)):
#     times = []
#     for i in range(100):
#         clf = models[clf_index]
#
#         t0 = time.time()
#         clf.fit(adult_data, adult_targets)
#         times.append(time.time() - t0)
#
#     print("{0} -> {1:10.5f}".format(model_names[clf_index], np.mean(times)))

r = statlog.load(encode_features=True)

# print(r.shape)
#
# r2 = r[~pd.isnull(r).any(axis=1)].astype(float)
#
# print(r2.shape)

print(r)