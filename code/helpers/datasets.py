import numpy as np
import pandas as pd

adult_data_path = "../datasets/adult/adult.data.txt"
adult_test_path = "../datasets/adult/adult.data.txt"


def load_adult_data():
    feature_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status", "Occupation",
                     "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country",
                     "Target"]

    adult_data = pd.read_csv(adult_data_path, names=feature_names, sep=r'\s*,\s*', engine='python', na_values="?")

    return adult_data

