import numpy as np
import pandas as pd

data_path = "../datasets/bank-marketing/bank-additional-full.txt"

feature_names = ["age", "job", "marital",  "education",  "default",  "housing",  "loan",  "contact",  "month",
                 "day_of_week",  "duration",  "campaign",  "pdays",  "previous",  "poutcome",  "emp.var.rate",
                 "cons.price.idx",  "cons.conf.idx",  "euribor3m",  "nr.employed"]

feature_classes = {"job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
                           "self-employed", "services", "student", "technician", "unemployed"],
                   "marital": ["divorced", "married", "single"],
                   "education": ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course",
                                 "university.degree"],
                   "default": ["no", "yes"],
                   "housing": ["no", "yes"],
                   "loan": ["no", "yes"],
                   "contact": ["cellular", "telephone"],
                   "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                   "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
                   "poutcome": ["failure", "nonexistent", "success"]}

target_classes = ["no", "yes"]

protected_features = ["age"]


def load(encode_features=False, remove_missing_values=False, verbose=False):
    """
    Loads Bank Marketing dataset.
    :param encode_features: if True, encodes all categorical features to numerical categories from 0 to number of
                            classes, and a numpy matrix will be returned.
    :param remove_missing_values: removes examples with missing value in the dataset.
    :param verbose: if True, prints information about how many missing values each feature had.
    :return: panda DataFrame or numpy matrix of the Bank Marketing dataset.
    """

    features_w_target = feature_names + ["Target"]

    if not encode_features:
        return pd.read_csv(data_path, names=features_w_target, na_values='unknown', sep=';', engine='python',
                           verbose=verbose)

    encoders = {"job": lambda x: feature_classes["job"].index(x) if not pd.isnull(x) else x,
                "marital": lambda x: feature_classes["marital"].index(x) if not pd.isnull(x) else x,
                "education": lambda x: feature_classes["education"].index(x) if not pd.isnull(x) else x,
                "default": lambda x: feature_classes["default"].index(x) if not pd.isnull(x) else x,
                "housing": lambda x: feature_classes["housing"].index(x) if not pd.isnull(x) else x,
                "loan": lambda x: feature_classes["loan"].index(x) if not pd.isnull(x) else x,
                "contact": lambda x: feature_classes["contact"].index(x) if not pd.isnull(x) else x,
                "month": lambda x: feature_classes["month"].index(x) if not pd.isnull(x) else x,
                "day_of_week": lambda x: feature_classes["day_of_week"].index(x) if x is not None else x,
                "poutcome": lambda x: feature_classes["poutcome"].index(x) if x is not None else x,
                "Target": lambda x: target_classes.index(x) if not pd.isnull(x) else x}

    dataset = pd.read_csv(data_path, names=features_w_target, na_values='unknown', converters=encoders, sep=';',
                          engine='python', verbose=verbose)

    np_data = dataset.as_matrix()

    if not remove_missing_values:
        return np_data

    # Deleting rows with empty values
    np_data = np_data[~pd.isnull(np_data).any(axis=1)].astype(float)

    return np_data
