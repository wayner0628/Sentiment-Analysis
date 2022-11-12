import pandas as pd

df = pd.read_csv('Dataset/train_HW2dataset.csv')

label = df["Emotion"].values

idx = pd.Index(label)

cnt = idx.value_counts()

print(cnt)
