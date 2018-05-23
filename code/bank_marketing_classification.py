import numpy as np
from sklearn import svm
from sklearn import tree
from helpers.datasets import bank_marketing
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier


names = ["DT", "RF", "GNB", "GPC", "KNN", "LR", "MLP", "SVC"]

classifiers = {
    "DT": tree.DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "GNB": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "LR": LogisticRegression(),
    "MLP": MLPClassifier(max_iter=1000),
    "SVC": svm.SVC(),
    "GPC": GaussianProcessClassifier(1.0 * RBF(1.0)),
}

param_grids = {
    "DT": {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 5)},
    "RF": {'n_estimators': range(1, 200), 'max_depth': range(1, 25)},
    "KNN": {'n_neighbors': range(1, 100)},
    "LR": [{'penalty': ['l2'], 'C': np.arange(0.1, 10, 0.1), 'tol': [1e-2, 1e-3, 1e-4, 1e-7, 1e-8],
           'solver': ['lbfgs', 'liblinear', 'sag', 'saga']},
           {'penalty': ['l1'], 'C': np.arange(0.1, 10, 0.1), 'tol': [1e-2, 1e-3, 1e-4, 1e-7, 1e-8],
           'solver': ['liblinear', 'saga']}],
    "MLP": {
            'hidden_layer_sizes': [(4,), (7,), (10, ), (9, 4), (12, 5), (15, ), (10, 10), (18, 6), (10, )],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            'epsilon': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-8],
            'solver': ['lbfgs', 'sgd', 'adam']},
    "SVC": [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.01, 0.1, 1, 5, 10, 50, 100, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}],
}

# Loading the learning set
statlog_data = bank_marketing.load(encode_features=True, remove_missing_values=True, verbose=False)
X = statlog_data[:, 0:-1]
y = statlog_data[:, -1].reshape(-1, 1)

X = StandardScaler().fit_transform(X)

# Splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

y_train_grid = y_train.ravel()

test_preds = {}
for classifier in names:
    print("\n\n{}: ".format(classifier))
    if classifier in param_grids:
        clf = GridSearchCV(classifiers[classifier], param_grid=param_grids[classifier], n_jobs=6, cv=5, refit=True)
        clf.fit(X_train, y_train_grid)
        print("\tBest parameters set found on development set:")
        c = clf.best_params_
        print("\t\t", c)
    else:
        clf = classifiers[classifier]
        clf.fit(X_train, y_train_grid)

    # predicting
    test_pred = clf.predict(X_test)

    test_preds[classifier] = test_pred

    print("\tAccuracy is: {0:3.2f}%".format(accuracy_score(y_test, test_pred) * 100))
