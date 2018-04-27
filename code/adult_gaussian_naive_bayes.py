import numpy as np
from sklearn.decomposition import PCA
import helpers.datasets.adult as adult
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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

prior_on_class_0 = len(np.where(adult_targets == 0)) / len(adult_targets)
# # clf = MultinomialNB(class_prior=np.array([prior_on_class_0, 1 - prior_on_class_0]))
clf = make_pipeline(StandardScaler(), GaussianNB())
# clf = GaussianNB()
clf = clf.fit(adult_data, adult_targets)

# predicting
adult_test_preds = clf.predict(adult_test_features)

print("Accuracy is: {0:3.2f}%".format(accuracy_score(adult_test_targets, adult_test_preds) * 100))


# # Drawing the decision tree
# dot_data = tree.export_graphviz(clf, out_file=None,
#                                 feature_names=adult.feature_names,
#                                 filled=True, rounded=True,
#                                 special_characters=True,
#                                 max_depth=5)
# graph = gviz.Source(dot_data)
# graph.render("Adult")

#
# adult.get_accuracy_for_feature_subset(adult_test, adult_test_preds, adult_test_targets, "Race")
# adult.get_accuracy_for_feature_subset(adult_test, adult_test_preds, adult_test_targets, "Sex")
# adult.get_accuracy_for_feature_subset(adult_test, adult_test_preds, adult_test_targets, "Country")
# adult.get_accuracy_for_feature_subset(adult_test, adult_test_preds, adult_test_targets, "Age")


# Evaluating demographic parity and equality of opportunity
# adult_test = adult.load("testing", encode_features=True)
# adult_test = adult.to_numpy_array(adult_test, remove_missing_values=True)
# adult.evaluate_demographic_parity(adult_test, clf, "Race")
# adult.evaluate_demographic_parity(adult_test, clf, "Sex")
# adult.evaluate_demographic_parity(adult_test, clf, "Country")
# adult.evaluate_demographic_parity(adult_test, clf, "Age")
#
# adult.evaluate_equality_of_opportunity(adult_test, clf, "Race")
# adult.evaluate_equality_of_opportunity(adult_test, clf, "Sex")
# adult.evaluate_equality_of_opportunity(adult_test, clf, "Country")
# adult.evaluate_equality_of_opportunity(adult_test, clf, "Age")
