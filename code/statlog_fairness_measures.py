import numpy as np
import pickle
from sklearn import svm
from sklearn import tree
from helpers.datasets import statlog
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from helpers import fair_metrics

# names = ["DT", "RF", "GNB", "GPC", "KNN", "LR", "MLP", "SVC"]
names = ["DT", "RF", "GNB", "GPC", "KNN", "LR", "MLP", "SVC"]

classifiers = {
    "DT": tree.DecisionTreeClassifier(criterion='gini', max_depth=3),
    "RF": RandomForestClassifier(max_depth=15, n_estimators=98),
    "GNB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=8),
    "LR": LogisticRegression(C=0.1, penalty='l1', solver='liblinear', tol=0.01),
    "MLP": MLPClassifier(activation='identity', epsilon=0.001, hidden_layer_sizes=(10,), solver='sgd', tol=1e-05),
    "SVC": svm.SVC(C=5, gamma=0.01, kernel='rbf'),
    "GPC": GaussianProcessClassifier(1.0 * RBF(1.0))
}

# Loading the learning set
statlog_data = statlog.load(encode_features=True)
X = statlog_data[:, 0:-1]
y = statlog_data[:, -1].reshape(-1, 1)

# Splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

y_train_grid = y_train.ravel()

# this is only for evaluating DVPR and EOVR
test_data = np.concatenate((X_test, y_test), axis=1)

fair_measures = {'FPR': {}, 'FNR': {}, 'DPVR': {}, 'EOVR': {}}
for classifier in names:
    fair_measures['FPR'][classifier] = {}
    fair_measures['FNR'][classifier] = {}
    fair_measures['DPVR'][classifier] = {}
    fair_measures['EOVR'][classifier] = {}


for classifier in names:
    print("{}: ".format(classifier))
    clf = make_pipeline(StandardScaler(), classifiers[classifier])
    clf.fit(X_train, y_train_grid)

    # predicting
    test_pred = clf.predict(X_test)

    for p_feature in statlog.protected_features:
        fpr, fnr = fair_metrics.get_accuracy_for_feature_subset(X_test, test_pred, y_test, p_feature, statlog)
        fair_measures['FPR'][classifier][p_feature] = fpr
        fair_measures['FNR'][classifier][p_feature] = fnr

        dpvr = fair_metrics.evaluate_demographic_parity(test_data, clf, p_feature, statlog)
        fair_measures['DPVR'][classifier][p_feature] = dpvr

        eovr = fair_metrics.evaluate_equality_of_opportunity(test_data, clf, p_feature, statlog)
        fair_measures['EOVR'][classifier][p_feature] = eovr

with open('../results/statlog/fair-measures-1.pckl', 'wb') as f:
    pickle.dump(fair_measures, f, protocol=-1)
