import numpy as np
from sklearn import svm
from helpers.datasets import adult
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Loading the learning set
adult_data = adult.load("learning", encode_features=True)
adult_data = adult.to_numpy_array(adult_data, remove_missing_values=True)

# Separating to target and features
adult_targets = adult_data[:, -1]
adult_data = adult_data[:, 0:-1]

# param_grid = {
#     'hidden_layer_sizes': [(4,), (7,), (10, ), (7, 4), (12, 5), (12, ), (10, 10), (8, ), (9, )],
#     'activation': ['identity', 'logistic', 'tanh', 'relu'],
#     'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
#     'epsilon': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-8],
#     'solver': ['lbfgs', 'sgd', 'adam'],
# }

# Grid search on NN
# clf = make_pipeline(StandardScaler(), GridSearchCV(MLPClassifier(), param_grid=param_grid, n_jobs=5, cv=5, refit=True))
# clf.fit(adult_data, adult_targets)
# print("Best parameters set found on development set:\n")
# c = clf.named_steps.gridsearchcv.best_params_
# print(c)

clf = make_pipeline(StandardScaler(), MLPClassifier(activation='tanh', epsilon=0.001, hidden_layer_sizes=(10,),
                                                    solver='lbfgs', tol=1e-06))
clf.fit(adult_data, adult_targets)

# Loading the test set
adult_test = adult.load("testing", encode_features=True)
adult_test = adult.to_numpy_array(adult_test, remove_missing_values=True)

adult_test_targets = adult_test[:, -1]
adult_test_features = adult_test[:, 0:-1]

# predicting
adult_test_preds = clf.predict(adult_test_features)


print("Accuracy is: {0:3.2f}%".format(accuracy_score(adult_test_targets, adult_test_preds) * 100))

# Evaluating demographic parity and equality of opportunity
adult_test = adult.load("testing", encode_features=True)
adult_test = adult.to_numpy_array(adult_test, remove_missing_values=True)

adult.evaluate_fairness(adult_test, adult_test_preds, adult_test_targets, clf, adult)


# {'activation': 'tanh', 'epsilon': 0.001, 'hidden_layer_sizes': (10,), 'solver': 'lbfgs', 'tol': 1e-06}
# Accuracy is: 84.97%