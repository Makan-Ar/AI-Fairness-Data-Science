import numpy as np
import pandas as pd
import sklearn.metrics as metric

data_path = "../datasets/adult/adult.data.txt"
test_path = "../datasets/adult/adult.test.txt"

feature_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status", "Occupation",
                 "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"]

feature_classes = {"Workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                                 "State-gov", "Without-pay", "Never-worked"],
                   "Education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
                                 "Assoc-voc",
                                 "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th",
                                 "Preschool"],
                   "Martial Status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                                      "Married-spouse-absent", "Married-AF-spouse"],
                   "Occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                                  "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                                  "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                                  "Armed-Forces"],
                   "Relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
                   "Race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
                   "Sex": ["Female", "Male"],
                   "Country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                               "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba",
                               "Iran",
                               "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico",
                               "Portugal",
                               "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
                               "Columbia",
                               "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
                               "El-Salvador",
                               "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]}

target_classes = [">50K", "<=50K"]


def load(path, encode_labels=False, verbose=False):
    features_w_target = feature_names + ['Target']

    if not encode_labels:
        return pd.read_csv(path, names=features_w_target, sep=r'\s*,\s*', engine='python',
                           na_values="?", verbose=verbose)

    encoders = {"Workclass": lambda x: feature_classes['Workclass'].index(x) if not pd.isnull(x) else x,
                "Education": lambda x: feature_classes['Education'].index(x) if not pd.isnull(x) else x,
                "Martial Status": lambda x: feature_classes['Martial Status'].index(x) if not pd.isnull(x) else x,
                "Occupation": lambda x: feature_classes['Occupation'].index(x) if not pd.isnull(x) else x,
                "Relationship": lambda x: feature_classes['Relationship'].index(x) if not pd.isnull(x) else x,
                "Race": lambda x: feature_classes['Race'].index(x) if not pd.isnull(x) else x,
                "Sex": lambda x: feature_classes['Sex'].index(x) if not pd.isnull(x) else x,
                "Country": lambda x: feature_classes['Country'].index(x) if x is not None else x,
                "Target": lambda x: target_classes.index(x.replace('.', '')) if not pd.isnull(x) else x}

    adult_data = pd.read_csv(path, names=features_w_target, converters=encoders, sep='\s*,\s*',
                             engine='python', na_values="?", verbose=verbose)

    return adult_data


def to_numpy_array(panda_data_frame, remove_missing_values=False):
    np_data = panda_data_frame.as_matrix()

    if not remove_missing_values:
        return np_data

    # Deleting rows with empty values
    np_data = np_data[~pd.isnull(np_data).any(axis=1)].astype(int)

    return np_data


def get_accuracy_for_feature_subset(data, y_pred, y_true, feature, subsets=None):
    """
    Prints the accuracy of the prediction of a subset(s) of a feature(s). For Adult data only.

    :param data: The dataset that was classified
    :param y_pred: predictions of the classifier
    :param y_true: true classes
    :param feature: the name of the feature to be used. Can be a single feature or a list of them.
    :param subsets: the subset of the feature above. Can be a single subset, a list of subsets. If not specified all
                   subsets will be evaluated.
    :return: None
    """

    if feature not in feature_names:
        print("Feature not found.")
        return

    feature_index = feature_names.index(feature)
    n = data.shape[0]

    # make sure subsets exists
    if subsets is None:
        subsets = feature_classes[feature]
    elif type(subsets) is str:
        subsets = list([subsets])
    elif type(subsets) is list:
        for sub in subsets:
            if sub not in feature_classes[feature]:
                print('Subset "{0}" not found in the {1}'.format(sub, feature))
                return

    print("\n{0} subset accuracy break down".format(feature))

    for subset in subsets:
        sub_index = feature_classes[feature].index(subset)

        # print(data[feature_index])
        subset_indices = np.where(data[:, feature_index] == sub_index)[0]
        subset_len = len(subset_indices)
        subset_proportion = 100 * subset_len / n

        subset_accuracy = metric.accuracy_score(y_true[subset_indices], y_pred[subset_indices]) * 100
        subset_precision = metric.precision_score(y_true[subset_indices], y_pred[subset_indices]) * 100

        subset_confusion_matrix = metric.confusion_matrix(y_true[subset_indices], y_pred[subset_indices])
        subset_false_negative_rate = subset_confusion_matrix[1, 0] / subset_len
        subset_false_positive_rate = subset_confusion_matrix[0, 1] / subset_len

        print("\t{0} -> Accuracy: {1:3.2f}% - Precision: {2:3.2f}% - FNR: {3:3.2f}% - FPR: {4:3.2f}% - Proportion: {5:3.2f}%"
              .format(subset, subset_accuracy, subset_precision, subset_false_negative_rate, subset_false_positive_rate, subset_proportion))

    return
