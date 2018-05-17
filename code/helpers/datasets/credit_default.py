import numpy as np
import pandas as pd

data_path = "../datasets/credit-default/credit-default.txt"

feature_names = ["LIMIT_BAL", "Sex", "Education", "Marriage", "Age", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5",
                 "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1",
                 "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

# Link to dataset -> https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
# Gender (1 = male; 2 = female).
# Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
# Marital status (1 = married; 2 = single; 3 = others)
feature_classes = {"Sex": ['1', '2'],
                   "Education": ['0', '1', '2', '3', '4', '5', '6'],
                   "Marriage": ['0', '1', '2', '3']}

# 1 is default and 0 is not.
target_classes = ['1', '0']

protected_features = ["Sex", "Age"]


def load(encode_features=False):
    """
    Loads Credit Default dataset.
    :param encode_features: if True, encodes all categorical features to numerical categories from 0 to number of
                            classes, and a numpy matrix will be returned.
    :return: panda DataFrame or numpy matrix of the Credit Default dataset.
    """

    features_w_target = feature_names + ["Target"]

    if not encode_features:
        return pd.read_csv(data_path, names=features_w_target, sep=',', engine='python')

    encoders = {"Sex": lambda x: feature_classes["Sex"].index(x) if not pd.isnull(x) else x,
                "Education": lambda x: feature_classes["Education"].index(x) if not pd.isnull(x) else x,
                "Marriage": lambda x: feature_classes["Marriage"].index(x) if not pd.isnull(x) else x,
                "Target": lambda x: target_classes.index(x.replace(".", "")) if not pd.isnull(x) else x}

    dataset = pd.read_csv(data_path, names=features_w_target, converters=encoders, sep=',', engine='python')
    return dataset.as_matrix().astype(float)
