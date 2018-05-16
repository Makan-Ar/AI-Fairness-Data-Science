import numpy as np
from sklearn.ensemble import RandomForestClassifier
from helpers.datasets import adult
from sklearn.metrics import accuracy_score

# Loading the learning set
adult_data = adult.load("learning", encode_features=True)
adult_data = adult.to_numpy_array(adult_data, remove_missing_values=True)

# Separating to target and features
adult_targets = adult_data[:, -1]
adult_data = adult_data[:, 0:-1]


# Loading the test set
adult_test = adult.load("testing", encode_features=True)
adult_test = adult.to_numpy_array(adult_test, remove_missing_values=True)

adult_test_targets = adult_test[:, -1]
adult_test_features = adult_test[:, 0:-1]

# Training the classifier
clf = RandomForestClassifier(n_estimators=50, max_features=None, max_depth=14)
clf = clf.fit(adult_data, adult_targets)

# predicting
adult_test_preds = clf.predict(adult_test_features)

print("Accuracy is: {0:3.4f}".format(accuracy_score(adult_test_targets, adult_test_preds)))

adult.evaluate_fairness(adult_test, adult_test_preds, adult_test_targets, clf, adult)