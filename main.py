import pandas as pd
import tensorflow as tf
from keras import layers, losses
from keras.layers import TextVectorization, Layer


label_map = {"neutral": 0, "anger": 1, "joy": 2, "surprise": 3, "sadness": 4, "disgust": 5, "fear": 6}


class ArgmaxLayer(Layer):
    def __init__(self):
        super(ArgmaxLayer, self).__init__()

    def call(self, inputs):
        pred = tf.math.argmax(inputs, axis=1)
        return tf.one_hot(pred, 7)


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
            max_tokens=self.max_words, output_mode="int", output_sequence_length=self.max_len, standardize="lower"
        )
        vectorize_layer.adapt(self.x_train)

        return vectorize_layer(self.x_train)


if __name__ == "__main__":
    preprocess = Preprocessing()
    preprocess.load_data()
    token = preprocess.prepare_tokens()
    label = tf.convert_to_tensor([label_map[label] for label in preprocess.y_train])
    # one_hot_label = tf.one_hot(label, 7)

    ds = tf.data.Dataset.from_tensor_slices((token, label))
    ds = ds.batch(32)
    # AUTOTUNE = tf.data.AUTOTUNE
    # ds = ds.cache().prefetch(buffer_size=AUTOTUNE)

    embedding_dim = 16

    model = tf.keras.Sequential(
        [
            layers.Embedding(preprocess.max_words + 1, embedding_dim),
            layers.Bidirectional(layers.LSTM(20, recurrent_activation="gelu")),
            layers.Dropout(0.1),
            # layers.Conv1D(60, 8, activation='gelu'),
            # layers.GlobalAveragePooling1D(),
            layers.Dense(20, activation="gelu"),
            layers.Dense(7, activation="gelu"),
            layers.Softmax(),
        ]
    )

    # f1 = tfa.metrics.F1Score(7, "macro")
    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=["accuracy"])

    epochs = 50
    model.fit(ds, epochs=epochs)

    model.save("saved_model/my_model")
