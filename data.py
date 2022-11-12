import pandas as pd

label_map = {"neutral": 0, "anger": 1, "joy": 2, "surprise": 3, "sadness": 4, "disgust": 5, "fear": 6}
cnt = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

df = pd.read_csv("Dataset/train_HW2dataset.csv")

text = df["Utterance"].values
label = df["Emotion"].values

for t, l in text, label:
    c = cnt[label_map[l]]
    if (c < 338):
        cnt[label_map[l]] += 1
    else:
        


