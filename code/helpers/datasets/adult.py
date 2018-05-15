import numpy as np
import pandas as pd

learning_data_path = "../datasets/adult/adult.data.txt"
testing_data_path = "../datasets/adult/adult.test.txt"
combined_data_path = "../datasets/adult/adult.combined.txt"

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
                               "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico",
                               "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan",
                               "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
                               "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]}

target_classes = ["<=50K", ">50K"]


def load(dataset, encode_features=False, verbose=False):
    """
    Loads UCI Adult dataset.
    :param dataset: options -> "learning": learning set only, "testing" testing set only, "both": both learning and
                    testing sets combined.
    :param encode_features: if True, encodes all categorical features to numerical categories from 0 to number of
                            classes.
    :param verbose: if True, prints information about how many missing values each feature had.
    :return: panda DataFrame of the Adult dataset.
    """

    if dataset == "learning":
        path = learning_data_path
    elif dataset == "testing":
        path = testing_data_path
    elif dataset == "both":
        path = combined_data_path
    else:
        print('Type of Adult dataset not determined. Choose "learning", "testing" or "both".')
        return

    features_w_target = feature_names + ['Target']

    if not encode_features:
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
    np_data = np_data[~pd.isnull(np_data).any(axis=1)].astype(float)

    return np_data

