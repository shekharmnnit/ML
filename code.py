import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file_path = 'https://raw.githubusercontent.com/shekharmnnit/ML/main/Customer%20Churn%20Dataset/customer_churn_dataset-training-master.csv'

training_df = pd.read_csv(file_path)
print(training_df.head(5).to_string())
print(training_df.isna().sum())