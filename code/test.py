import pandas as pd
import helpers.datasets as db

adult_data = db.load_adult_data()
print(type(adult_data.as_matrix()))