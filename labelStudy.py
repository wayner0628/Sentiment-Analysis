import pandas as pd

df = pd.read_csv('Dataset/dev_HW2dataset.csv')
# df = pd.read_csv('Dataset/Adjust_dataset.csv')

label = df["Emotion"].values

idx = pd.Index(label)

cnt = idx.value_counts()

print(cnt)
