import numpy as np
import pandas as pd
import helpers.datasets as db

# adult_data = db.load_adult_data()
# adult_test_data = db.load_adult_test()

###### Checking which features have missing values #####
# print(adult_data.isnull().any())

adult_data = db.load_adult(db.adult_test_path, encode_labels=True)

# print(adult_data['Education'])

# print(adult_data.head(15))
# m = adult_data.as_matrix()

print(adult_data.head(10))
# print(type(m[14, -2]))