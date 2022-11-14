import pandas as pd

df = pd.read_csv('Dataset/Adjust_train_dataset.csv')
# df = pd.read_csv('Dataset/Adjust_dataset.csv')

label = df["Emotion"].values

idx = pd.Index(label)

cnt = idx.value_counts()

print(cnt)

text = df['Utterance'].values

s = set()

for words in text:
    words = words.split()
    for word in words:
        if word not in s:
            s.add(word)
            # print(word)

print(len(s))
