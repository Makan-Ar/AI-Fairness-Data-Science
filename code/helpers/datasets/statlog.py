import numpy as np
import pandas as pd
import sklearn.metrics as metric

learning_data_path = "../datasets/Statlog/german.data.txt"

feature_names = ["checking-status", "duration", "credit-history", "purpose", "credit-amount", "saving-account",
                 "present-employment-duration", "installment-income-ratio", "sex-marital-status", "other-debtor",
                 "present-residence-duration", "property", "age", "other-installments", "housing", "existing-credits",
                 "job", "num-people-liable", "telephone", "foreign-worker"]


# info on attr classes -> https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29

# Attribute Information:
#
# Attribute 1: (qualitative)
# Status of existing checking account
# A11 : ... < 0 DM
# A12 : 0 <= ... < 200 DM
# A13 : ... >= 200 DM / salary assignments for at least 1 year
# A14 : no checking account
#
# Attribute 2: (numerical)
# Duration in month
#
# Attribute 3: (qualitative)
# Credit history
# A30 : no credits taken/ all credits paid back duly
# A31 : all credits at this bank paid back duly
# A32 : existing credits paid back duly till now
# A33 : delay in paying off in the past
# A34 : critical account/ other credits existing (not at this bank)
#
# Attribute 4: (qualitative)
# Purpose
# A40 : car (new)
# A41 : car (used)
# A42 : furniture/equipment
# A43 : radio/television
# A44 : domestic appliances
# A45 : repairs
# A46 : education
# A47 : (vacation - does not exist?)
# A48 : retraining
# A49 : business
# A410 : others
#
# Attribute 5: (numerical)
# Credit amount
#
# Attibute 6: (qualitative)
# Savings account/bonds
# A61 : ... < 100 DM
# A62 : 100 <= ... < 500 DM
# A63 : 500 <= ... < 1000 DM
# A64 : .. >= 1000 DM
# A65 : unknown/ no savings account
#
# Attribute 7: (qualitative)
# Present employment since
# A71 : unemployed
# A72 : ... < 1 year
# A73 : 1 <= ... < 4 years
# A74 : 4 <= ... < 7 years
# A75 : .. >= 7 years
#
# Attribute 8: (numerical)
# Installment rate in percentage of disposable income
#
# Attribute 9: (qualitative)
# Personal status and sex
# A91 : male : divorced/separated
# A92 : female : divorced/separated/married
# A93 : male : single
# A94 : male : married/widowed
# A95 : female : single
#
# Attribute 10: (qualitative)
# Other debtors / guarantors
# A101 : none
# A102 : co-applicant
# A103 : guarantor
#
# Attribute 11: (numerical)
# Present residence since
#
# Attribute 12: (qualitative)
# Property
# A121 : real estate
# A122 : if not A121 : building society savings agreement/ life insurance
# A123 : if not A121/A122 : car or other, not in attribute 6
# A124 : unknown / no property
#
# Attribute 13: (numerical)
# Age in years
#
# Attribute 14: (qualitative)
# Other installment plans
# A141 : bank
# A142 : stores
# A143 : none
#
# Attribute 15: (qualitative)
# Housing
# A151 : rent
# A152 : own
# A153 : for free
#
# Attribute 16: (numerical)
# Number of existing credits at this bank
#
# Attribute 17: (qualitative)
# Job
# A171 : unemployed/ unskilled - non-resident
# A172 : unskilled - resident
# A173 : skilled employee / official
# A174 : management/ self-employed/
# highly qualified employee/ officer
#
# Attribute 18: (numerical)
# Number of people being liable to provide maintenance for
#
# Attribute 19: (qualitative)
# Telephone
# A191 : none
# A192 : yes, registered under the customers name
#
# Attribute 20: (qualitative)
# foreign worker
# A201 : yes
# A202 : no

feature_classes = {"checking-status": ["A11", "A12", "A13", "A14"],
                   "credit-history": ["A30", "A31", "A32", "A33", "A34"],
                   "purpose": ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"],
                   "saving-account": ["A61", "A62", "A63", "A64", "A65"],
                   "present-employment-duration": ["A71", "A72", "A73", "A74", "A75"],
                   "sex-marital-status": ["A91", "A92", "A93", "A94", "A95"],
                   "other-debtor": ["A101", "A102", "A103"],
                   "property": ["A121", "A122", "A123", "A124"],
                   "other-installments": ["A141", "A142", "A143"],
                   "housing": ["A151", "A152", "A153"],
                   "job": ["A171", "A172", "A173", "A174"],
                   "telephone": ["A191", "A191"],
                   "foreign-worker": ["A201", "A202"]}

target_classes = ["1", "2"]

