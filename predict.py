from main import Preprocessing
import pandas as pd
import numpy as np
import tensorflow as tf


preprocess = Preprocessing()
preprocess.load_data()

df = pd.read_csv('Dataset/test_HW2dataset.csv')
text = df["Utterance"].values

token = preprocess.prepare_tokens(text)

new_model = tf.keras.models.load_model("saved_model/my_model")
new_model.summary()

res = new_model.predict(token)
# print(res)
res = tf.math.argmax(res, axis=1)
res = res.numpy()

index = np.arange(3400)

data = {"index": index, "emotion": res}

DF = pd.DataFrame.from_dict(data)
DF.to_csv("output.csv", index=False)
