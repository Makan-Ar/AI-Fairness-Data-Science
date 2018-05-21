import pickle
import numpy as np
from sklearn import svm
from sklearn import tree
from helpers import fair_metrics
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from helpers.datasets import credit_default
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier

# names = ["DT", "RF", "GNB", "GPC", "KNN", "LR", "MLP", "SVC"]
names = ["DT", "RF", "GNB", "GPC", "KNN", "LR", "MLP", "SVC"]

classifiers = {
    "DT": tree.DecisionTreeClassifier(criterion='gini', max_depth=2),
    "RF": RandomForestClassifier(max_depth=11, n_estimators=52),
    "GNB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=29),
    "LR": LogisticRegression(C=0.1, penalty='l1', solver='liblinear', tol=0.01),
    "MLP": MLPClassifier(activation='logistic', epsilon=1e-08, hidden_layer_sizes=(4,), solver='lbfgs', tol=1e-07),
    "SVC": svm.SVC(C=10, gamma=0.01, kernel='rbf'),
    "GPC": GaussianProcessClassifier(1.0 * RBF(1.0))
}

# Loading the learning set
credit_default_data = credit_default.load(encode_features=True)
X = credit_default_data[:, 0:-1]
y = credit_default_data[:, -1].reshape(-1, 1)

# Splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

y_train_grid = y_train.ravel()

# this is only for evaluating DVPR and EOVR
test_data = np.concatenate((X_test, y_test), axis=1)

fair_measures = {'FPR': {}, 'FNR': {}, 'DPVR': {}, 'EOVR': {}}
pred_accuracy = {}

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
    pred_accuracy[classifier] = accuracy_score(y_test, test_pred) * 100

    for p_feature in credit_default.protected_features:
        fpr, fnr = fair_metrics.get_accuracy_for_feature_subset(X_test, test_pred, y_test, p_feature, credit_default)
        fair_measures['FPR'][classifier][p_feature] = fpr
        fair_measures['FNR'][classifier][p_feature] = fnr

        dpvr = fair_metrics.evaluate_demographic_parity(test_data, clf, p_feature, credit_default)
        fair_measures['DPVR'][classifier][p_feature] = dpvr

        eovr = fair_metrics.evaluate_equality_of_opportunity(test_data, clf, p_feature, credit_default)
        fair_measures['EOVR'][classifier][p_feature] = eovr

with open('../results/credit-default/fair-measures-1.pckl', 'wb') as f:
    pickle.dump(fair_measures, f, protocol=-1)

with open('../results/credit-default/pred-accuracy-1.pckl', 'wb') as f:
    pickle.dump(pred_accuracy, f, protocol=-1)
