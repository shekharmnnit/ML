import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

trainDatasetfilePath = 'https://raw.githubusercontent.com/shekharmnnit/ML/main/Customer%20Churn%20Dataset/customer_churn_dataset-training-master.csv'
testDatasetPath= 'https://raw.githubusercontent.com/shekharmnnit/ML/main/Customer%20Churn%20Dataset/customer_churn_dataset-testing-master.csv'
training_df = pd.read_csv(trainDatasetfilePath)
testing_df = pd.read_csv(testDatasetPath)
print(training_df.head(5).to_string())
print(training_df.isna().sum())

print(testing_df.head(5).to_string())
print(testing_df.isna().sum())
