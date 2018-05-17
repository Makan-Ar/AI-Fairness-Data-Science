import pickle
from sklearn import svm
from sklearn import tree
from helpers import fair_metrics
from helpers.datasets import adult
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

# names = ["DT", "RF", "GNB", "GPC", "KNN", "LR", "MLP", "SVC"]
names = ["DT", "RF", "GNB", "GPC", "KNN", "LR", "MLP", "SVC"]

classifiers = {
    "DT": tree.DecisionTreeClassifier(max_depth=3, criterion="entropy"),
    "RF": RandomForestClassifier(n_estimators=50, max_features=None, max_depth=14),
    "GNB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "LR": LogisticRegression(C=6.6, penalty='l1', tol=0.01),
    "MLP": MLPClassifier(activation='tanh', epsilon=0.001, hidden_layer_sizes=(10,), solver='lbfgs', tol=1e-06),
    "SVC": svm.SVC(kernel="rbf", C=100, gamma=0.001),
    "GPC": GaussianProcessClassifier(1.0 * RBF(1.0))
}

# Loading the learning set
train_data = adult.load("learning", encode_features=True)
train_data = adult.to_numpy_array(train_data, remove_missing_values=True)
X_train = train_data[:, 0:-1]
y_train = train_data[:, -1]

# Loading the test set
test_data = adult.load("testing", encode_features=True)
test_data = adult.to_numpy_array(test_data, remove_missing_values=True)
X_test = test_data[:, 0:-1]
y_test = test_data[:, -1]

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
    clf.fit(X_train, y_train)

    # predicting
    test_pred = clf.predict(X_test)
    pred_accuracy[classifier] = accuracy_score(y_test, test_pred) * 100

    for p_feature in adult.protected_features:
        fpr, fnr = fair_metrics.get_accuracy_for_feature_subset(X_test, test_pred, y_test, p_feature, adult)
        fair_measures['FPR'][classifier][p_feature] = fpr
        fair_measures['FNR'][classifier][p_feature] = fnr

        dpvr = fair_metrics.evaluate_demographic_parity(test_data, clf, p_feature, adult)
        fair_measures['DPVR'][classifier][p_feature] = dpvr

        eovr = fair_metrics.evaluate_equality_of_opportunity(test_data, clf, p_feature, adult)
        fair_measures['EOVR'][classifier][p_feature] = eovr

with open('../results/adult/fair-measures-1.pckl', 'wb') as f:
    pickle.dump(fair_measures, f, protocol=-1)

with open('../results/adult/pred-accuracy-1.pckl', 'wb') as f:
    pickle.dump(pred_accuracy, f, protocol=-1)
