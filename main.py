import pandas as pd
import tensorflow as tf
from keras import layers, losses
from keras.layers import TextVectorization


label_map = {"neutral": 0, "anger": 1, "joy": 2, "surprise": 3, "sadness": 4, "disgust": 5, "fear": 6}


class Preprocessing:
    def __init__(self):
        self.data = "Dataset/train_HW2dataset.csv"
        self.max_len = 30
        self.max_words = 10000

    def load_data(self):
        df = pd.read_csv(self.data)
        df = df.drop(
            [
                "Speaker",
                "Dialogue_ID",
                "Utterance_ID",
                "Old_Dialogue_ID",
                "Old_Utterance_ID",
                "Season",
                "Episode",
                "StartTime",
                "EndTime",
            ],
            axis=1,
            inplace=False,
        )

        X = df["Utterance"].values
        Y = df["Emotion"].values

        self.x_train, self.y_train = X, Y
        return df

    def prepare_tokens(self):
        vectorize_layer = TextVectorization(
            max_tokens=self.max_words, output_mode="int", output_sequence_length=self.max_len
        )
        vectorize_layer.adapt(self.x_train)

        return vectorize_layer(self.x_train)


if __name__ == "__main__":
    preprocess = Preprocessing()
    preprocess.load_data()
    token = preprocess.prepare_tokens()
    label = tf.convert_to_tensor([label_map[label] for label in preprocess.y_train])

    ds = tf.data.Dataset.from_tensor_slices((token, label))
    ds = ds.batch(32)

    # AUTOTUNE = tf.data.AUTOTUNE
    # ds = ds.cache().prefetch(buffer_size=AUTOTUNE)

    embedding_dim = 16
    model = tf.keras.Sequential(
        [
            layers.Embedding(preprocess.max_words + 1, embedding_dim),
            layers.Conv1D(60, 8, activation='gelu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(20, activation="gelu"),
            layers.Dense(7, activation="gelu"),
            layers.Dropout(0.1),
            layers.Softmax(),
        ]
    )

    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=["accuracy"])

    epochs = 100
    history = model.fit(ds, epochs=epochs)
    history_dict = history.history
    history_dict.keys()

    acc = history_dict["accuracy"]
    loss = history_dict["loss"]

    epochs = range(1, len(acc) + 1)

    # examples = ["My duties?  All right.", "No don’t I beg of you!", "I see."]

    model.save("saved_model/my_model")
