import pandas as pd
import re

label_map = {"neutral": 0, "anger": 1, "joy": 2, "surprise": 3, "sadness": 4, "disgust": 5, "fear": 6}
cnt = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

df = pd.read_csv("Dataset/dev_HW2dataset.csv")

text = df["Utterance"].values
idx = 0

for i in range(df.shape[0]):
    t = text[i]
    t = re.sub("[?]", " ?", t)
    t = re.sub("!", " !", t)
    t = re.sub("[.][.][.]", " #", t)
    t = re.sub("-", " - ", t)
    t = re.sub(",", " ", t)
    t = re.sub("’", " ’", t)
    t = re.sub("[.]", " .", t)
    t = re.sub("#", "...", t)
    t = re.sub(":", " ", t)
    t = re.sub('"', " ", t)
    t = re.sub("‘", " ", t)
    t = re.sub("'", " ", t)
    t = re.sub("…", " … ", t)
    # l = label[i]
    # c = cnt[label_map[l]]
    # if c < 338:
    #     df["Utterance"][idx] = t
    #     cnt[label_map[l]] += 1
    #     idx += 1
    # else:
    #     df = df.drop(
    #         [i],
    #         axis=0,
    #         inplace=False,
    #     )
    df["Utterance"][i] = t

df.to_csv("./Dataset/Adjust_dev_dataset.csv")
