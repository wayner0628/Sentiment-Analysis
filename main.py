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
    def __init__(self, path="Dataset/train_HW2dataset.csv"):
        # self.data = "Dataset/train_HW2dataset.csv"
        self.data = path
        self.max_len = 50
        self.max_words = 50000

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

    def prepare_tokens(self, text):
        vectorize_layer = TextVectorization(
            max_tokens=self.max_words, output_mode="int", output_sequence_length=self.max_len, standardize="lower"
        )
        vectorize_layer.adapt(self.x_train)

        return vectorize_layer(text)


if __name__ == "__main__":
    preprocess = Preprocessing()
    preprocess.load_data()
    token = preprocess.prepare_tokens(preprocess.x_train)
    label = tf.convert_to_tensor([label_map[label] for label in preprocess.y_train])
    # one_hot_label = tf.one_hot(label, 7)

    val_preprocess = Preprocessing(path="Dataset/Adjust_dev_dataset.csv")
    val_preprocess.load_data()
    val_token = preprocess.prepare_tokens(val_preprocess.x_train)
    val_label = tf.convert_to_tensor([label_map[label] for label in val_preprocess.y_train])

    ds = tf.data.Dataset.from_tensor_slices((token, label))
    ds = ds.batch(32)
    # AUTOTUNE = tf.data.AUTOTUNE
    # ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((val_token, val_label))
    val_ds = val_ds.batch(32)

    embedding_dim = 128

    model = tf.keras.Sequential(
        [
            layers.Embedding(preprocess.max_words + 1, embedding_dim),
            layers.Bidirectional(layers.LSTM(60)),
            # layers.Dropout(0.1),
            # layers.Conv1D(60, 8, activation='gelu'),
            # layers.GlobalAveragePooling1D(),
            layers.Dense(80),
            layers.Dense(50),
            layers.Dense(20),
            layers.Dense(7, activation="gelu"),
            layers.Softmax(),
        ]
    )

    # f1 = tfa.metrics.F1Score(7, "macro")
    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=["accuracy"])

    epochs = 30
    model.fit(ds, epochs=epochs, validation_data=val_ds)

    model.save("saved_model/my_model")
