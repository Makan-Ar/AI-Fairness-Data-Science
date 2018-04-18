import numpy as np
import pandas as pd
import graphviz as gviz
from sklearn import tree
import helpers.datasets.adult as adult
from sklearn.metrics import accuracy_score


adult_data = adult.load('learning', encode_features=True)
adult_data = adult.to_numpy_array(adult_data, remove_missing_values=True)

adult.print_feature_subsets_proportions(adult_data, "Country")