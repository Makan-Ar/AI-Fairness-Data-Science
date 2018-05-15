import numpy as np
from sklearn import svm
import helpers.datasets.statlog as statlog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# Loading the learning set
statlog_data = statlog.load(encode_features=True)
X = statlog_data[:, 0:-1]
y = statlog_data[:, -1].reshape(-1, 1)

# Splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

y_train_grid = y_train.ravel()

# # Grid search on SVC
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.01, 0.1, 1, 5, 10, 50, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


clf = make_pipeline(StandardScaler(), GridSearchCV(svm.SVC(), param_grid=param_grid, n_jobs=5, cv=5, refit=True))
clf.fit(X_train, y_train_grid)
print("Best parameters set found on development set:\n")
c = clf.named_steps.gridsearchcv.best_params_
print(c)


# Training the classifier
clf = svm.SVC()
clf = clf.fit(X_train, y_train)

# predicting
test_preds = clf.predict(X_test)


print("Accuracy is: {0:3.2f}%".format(accuracy_score(y_test, test_preds) * 100))
print("F-1 score is: {0:3.2f}%".format(f1_score(y_test, test_preds) * 100))
