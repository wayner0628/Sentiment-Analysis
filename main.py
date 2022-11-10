import pandas as pd
import tensorflow as tf

df = pd.read_csv("./train_HW2dataset.csv")

entry_size = df.shape[0]
sentences = []
labels = []


for entry in range(entry_size):
    sentences.append(df["Utterance"][entry])

# sentences = tf.convert_to_tensor([1, 1])
# print(sentences)
