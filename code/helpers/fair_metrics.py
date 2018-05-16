import numpy as np
import sklearn.metrics as metric
from helpers.datasets import adult
from helpers.datasets import statlog

age_subsets = [(0, 17), (18, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 200)]


def print_feature_subsets_proportions(data, feature, dataset):

    if feature not in dataset.feature_names:
        print("Feature not found.")
        return

    feature_index = dataset.feature_names.index(feature)
    n = data.shape[0]

    # make sure subsets exists
    if feature.lower() == "age":
        subsets = age_subsets
    else:
        subsets = dataset.feature_classes[feature]

    print("\n{0} subset proportion break down".format(feature))

    for subset in subsets:
        if feature.lower() == "age":
            subset_indices = np.where(np.logical_and(data[:, feature_index] >= subset[0],
                                                     data[:, feature_index] <= subset[1]))[0]
        else:
            sub_index = dataset.feature_classes[feature].index(subset)
            subset_indices = np.where(data[:, feature_index] == sub_index)[0]

        subset_len = len(subset_indices)
        if subset_len == 0:
            print("\t{0} -> None exists.".format(subset))
            continue

        subset_positive = len(np.where(data[subset_indices, -1] == 1)[0]) / n
        subset_negative = len(np.where(data[subset_indices, -1] == 0)[0]) / n

        print("\t {0}\t & {1:2.3f} & {2:2.3f}".format(subset, subset_negative, subset_positive))

    return


def get_accuracy_for_feature_subset(data, y_pred, y_true, feature, dataset, subsets=None):
    """
    Prints the accuracy of the prediction of a subset(s) of a feature(s).

    :param data: The dataset that was classified
    :param y_pred: predictions of the classifier
    :param y_true: true classes
    :param feature: the name of the feature to be used. Can be a single feature or a list of them.
    :param dataset: pointer to the dataset helper package.
    :param subsets: the subset of the feature above. Can be a single subset, a list of subsets. If not specified all
                   subsets will be evaluated. This is ignored on non-categorical features.
    :return: None
    """

    if feature not in dataset.feature_names:
        print("Feature not found.")
        return

    feature_index = dataset.feature_names.index(feature)
    n = data.shape[0]

    # make sure subsets exists
    if feature.lower() == "age":
        subsets = age_subsets
    elif subsets is None:
        subsets = dataset.feature_classes[feature]
    elif type(subsets) is str:
        subsets = list([subsets])
    elif type(subsets) is list:
        for sub in subsets:
            if sub not in dataset.feature_classes[feature]:
                # print('Subset "{0}" not found in the {1}'.format(sub, feature))
                return

    print("\n{0} subset accuracy break down".format(feature))
    all_subsets_fp = []
    all_subsets_fn = []
    for subset in subsets:
        if feature.lower() == "age":
            subset_indices = np.where(np.logical_and(data[:, feature_index] >= subset[0],
                                                     data[:, feature_index] <= subset[1]))[0]
        else:
            sub_index = dataset.feature_classes[feature].index(subset)
            subset_indices = np.where(data[:, feature_index] == sub_index)[0]

        subset_len = len(subset_indices)
        if subset_len == 0:
            # print("\t{0} -> None exists.".format(subset))
            continue

        subset_proportion = 100 * subset_len / n
        subset_y_true = y_true[subset_indices]
        subset_y_pred = y_pred[subset_indices]

        # subset_accuracy = metric.accuracy_score(subset_y_true, subset_y_pred) * 100
        # subset_precision = metric.precision_score(subset_y_true, subset_y_pred) * 100
        subset_confusion_matrix = metric.confusion_matrix(subset_y_true, subset_y_pred)

        if len(np.unique(subset_y_pred)) == 1:
            subset_false_negative_rate = -1
            subset_false_positive_rate = -1
            continue
        else:
            subset_false_negative_rate = subset_confusion_matrix[1, 0] / subset_len
            subset_false_positive_rate = subset_confusion_matrix[0, 1] / subset_len

        all_subsets_fn.append(subset_false_negative_rate)
        all_subsets_fp.append(subset_false_positive_rate)

        # print("\t{0} -> Accuracy: {1:3.2f}% - Precision: {2:3.2f}% - FNR: {3:3.2f}% - "
        #       "FPR: {4:3.2f}% - Proportion: {5:3.2f}%"
        #       .format(subset, subset_accuracy, subset_precision, subset_false_negative_rate,
        #               subset_false_positive_rate, subset_proportion))

    print("Average FPR:", np.mean(all_subsets_fp))
    print("Average FNR:", np.mean(all_subsets_fn))
    return


def evaluate_demographic_parity(data, clf, feature, dataset):
    if feature not in dataset.feature_names:
        print("Feature not found.")
        return

    feature_index = dataset.feature_names.index(feature)
    if data.shape[1] != len(dataset.feature_names) + 1:
        print("Dataset shape does not match!")
        return

    # make sure subsets exists
    if feature.lower() == "age":
        subsets = age_subsets
    else:
        subsets = dataset.feature_classes[feature]

    print("\n{0} subset demographic parity break down".format(feature))
    all_subsets_eo = []

    for subset in subsets:
        changed_indices = set()

        if feature.lower() == "age":
            subset_indices = np.where(np.logical_and(data[:, feature_index] >= subset[0],
                                                     data[:, feature_index] <= subset[1]))[0]
        else:
            sub_index = dataset.feature_classes[feature].index(subset)
            subset_indices = np.where(data[:, feature_index] == sub_index)[0]

        n_subset = len(subset_indices)

        if len(subset_indices) == 0:
            # print("\t {0} -> no instance found.".format(subset))
            continue

        subset_data = data[subset_indices, :]
        subset_test = subset_data[:, 0:-1]

        # predicting
        actual_preds = clf.predict(subset_test)

        for diff_subset in subsets:
            if diff_subset == subset:
                continue

            if feature.lower() == "age":
                if diff_subset[0] == 70:
                    average_subset_age = 75
                elif diff_subset[0] == 0:
                    average_subset_age = 16
                else:
                    average_subset_age = (diff_subset[0] + diff_subset[1]) / 2

                subset_test[:, feature_index] = subset_test[:, feature_index] * 0 + average_subset_age
            else:
                diff_sub_index = dataset.feature_classes[feature].index(diff_subset)

                # change the subset to another one
                subset_test[:, feature_index] = subset_test[:, feature_index] * 0 + diff_sub_index

            diff_sub_preds = clf.predict(subset_test)

            # Check which predictions changed by changing the class membership and adding them to the changed list
            for x in np.where(np.not_equal(actual_preds, diff_sub_preds))[0]:
                changed_indices.add(x)

        all_subsets_eo.append(len(changed_indices) / n_subset)
        # print("\t {0},{1:2.4f},{2},{3}".format(subset, len(changed_indices) / n_subset, len(changed_indices), n_subset))

    print("Average:", np.mean(all_subsets_eo))
    return


def evaluate_equality_of_opportunity(data, clf, feature, dataset):
    if feature not in dataset.feature_names:
        print("Feature not found.")
        return

    feature_index = dataset.feature_names.index(feature)
    if data.shape[1] != len(dataset.feature_names) + 1:
        print("Dataset shape does not match!")
        return

    # make sure subsets exists
    if feature.lower() == "age":
        subsets = age_subsets
    else:
        subsets = dataset.feature_classes[feature]

    print("\n{0} subset equality of opportunity break down".format(feature))
    all_subsets_eo = []

    for subset in subsets:
        changed_indices = set()

        # get indices of a particular subset which are labeled as 1 (True examples only)
        if feature.lower() == "age":
            subset_indices = np.where(np.logical_and(np.logical_and(data[:, feature_index] >= subset[0],
                                                     data[:, feature_index] <= subset[1]), data[:, -1] == 1))[0]
        else:
            sub_index = dataset.feature_classes[feature].index(subset)
            subset_indices = np.where(np.logical_and(data[:, feature_index] == sub_index, data[:, -1] == 1))[0]

        if len(subset_indices) == 0:
            # print("\t {0} -> no instance found.".format(subset))
            continue

        subset_data = data[subset_indices, :]
        subset_test = subset_data[:, 0:-1]

        # predicting
        actual_preds = clf.predict(subset_test)
        n_true_positive = len(np.where(actual_preds == 1)[0])

        if n_true_positive == 0:
            # print("\t {0} -> no prediction of >50K income".format(subset))
            continue

        for diff_subset in subsets:
            if diff_subset == subset:
                continue

            if feature.lower() == "age":
                if diff_subset[0] == 70:
                    average_subset_age = 75
                elif diff_subset[0] == 0:
                    average_subset_age = 16
                else:
                    average_subset_age = (diff_subset[0] + diff_subset[1]) / 2

                subset_test[:, feature_index] = subset_test[:, feature_index] * 0 + average_subset_age
            else:
                diff_sub_index = dataset.feature_classes[feature].index(diff_subset)

                # change the subset to another one
                subset_test[:, feature_index] = subset_test[:, feature_index] * 0 + diff_sub_index

            diff_sub_preds = clf.predict(subset_test)

            # Check which true positive predictions changed by changing the class membership
            for i in range(len(actual_preds)):
                if actual_preds[i] == 1 and actual_preds[i] != diff_sub_preds[i]:
                    changed_indices.add(i)

        all_subsets_eo.append(len(changed_indices) / n_true_positive)
        # print("\t {0},{1:2.4f},{2},{3}".format(subset, len(changed_indices) / n_true_positive,
        #                                        len(changed_indices), n_true_positive))

    print("Average:", np.mean(all_subsets_eo))
    return
