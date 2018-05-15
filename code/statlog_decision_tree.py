import numpy as np
# import graphviz as gviz
from sklearn import tree
import helpers.datasets.statlog as statlog
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Loading the learning set
statlog_data = statlog.load(encode_features=True)
X = statlog_data[:, 0:-1]
y = statlog_data[:, -1].reshape(-1, 1)

# Splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

# Training the classifier
clf = tree.DecisionTreeClassifier(max_depth=3, criterion="entropy")
clf = clf.fit(X_train, y_train)

# predicting
test_preds = clf.predict(X_test)


print("Accuracy is: {0:3.2f}%".format(accuracy_score(y_test, test_preds) * 100))
print("F-1 score is: {0:3.2f}%".format(f1_score(y_test, test_preds) * 100))
