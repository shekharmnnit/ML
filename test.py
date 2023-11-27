import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

trainDatasetfilePath = 'https://raw.githubusercontent.com/shekharmnnit/ML/main/Customer%20Churn%20Dataset/customer_churn_dataset-training-master.csv'
testDatasetPath= 'https://raw.githubusercontent.com/shekharmnnit/ML/main/Customer%20Churn%20Dataset/customer_churn_dataset-testing-master.csv'
training_df = pd.read_csv(trainDatasetfilePath)
testing_df = pd.read_csv(testDatasetPath)

print("----Training Data set----------")
print(training_df.head(5).to_string())
print(f"Number of observation in training dataset: {training_df.shape[0]}")
print("Null value count")
print(training_df.isna().sum())
print(f"Training: Total number of rows with null value = {training_df.isna().sum().sum()}")

print("----Testing Data set----------")
print(testing_df.head(5).to_string())
print(f"Number of observation in test dataset: {testing_df.shape[0]}")
print("Null value count")
print(testing_df.isna().sum())
print(f"Testing: Total number of rows with null value = {testing_df.isna().sum().sum()}")

print('-------------------Train Data cleaning-------------------')
print(training_df[training_df['Age'].isna()].to_string()) # row detail with na value
print("199295 row has null value for all the columns, so removing 199295")
training_df = training_df.drop(training_df[training_df['Age'].isna()].index)
print(training_df.isna().sum())

print(f"Train data duplicated= {training_df.duplicated().sum()}")
print(f"Test data duplicated= {testing_df.duplicated().sum()}")
print("no need to remove duplicate data as duplicate data is 0")