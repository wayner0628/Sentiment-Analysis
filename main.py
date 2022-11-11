import pandas as pd
import tensorflow as tf
import keras
from keras import layers, losses
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import TextVectorization, Layer


label_map = {"neutral": 0, "anger": 1, "joy": 2, "surprise": 3, "sadness": 4, "disgust": 5, "fear": 6}


class Attention(Layer):
    def __init__(
        self,
        step_dim,
        W_regularizer=None,
        b_regularizer=None,
        W_constraint=None,
        b_constraint=None,
        bias=True,
        **kwargs
    ):
        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
            (input_shape[-1],),
            initializer=self.init,
            name="{}_W".format(self.name),
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(
                (input_shape[1],),
                initializer="zero",
                name="{}_b".format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


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
    query = keras.Input(shape=(None,))
    value = keras.Input(shape=(None,))
    key = keras.Input(shape=(None,))

    embedding_dim = 16
    # token_embedding = layers.Embedding(preprocess.max_words + 1, embedding_dim)
    # cnn_layer = layers.Conv1D(
    #     filters=100,
    #     kernel_size=4,
    #     # Use 'same' padding so outputs have the same shape as inputs.
    #     padding="same",
    # )
    # query_embeddings = cnn_layer(token_embedding(query))
    # value_embeddings = cnn_layer(token_embedding(value))
    # key_embeddings = cnn_layer(token_embedding(key))

    model = tf.keras.Sequential(
        [
            layers.Embedding(preprocess.max_words + 1, embedding_dim),
            # layers.Attention()([query_embeddings, value_embeddings, key_embeddings]),
            layers.Bidirectional(layers.LSTM(20, recurrent_activation="gelu")),
            layers.Dropout(0.1),
            # layers.Conv1D(60, 8, activation='gelu'),
            # layers.GlobalAveragePooling1D(),
            layers.Dense(20, activation="gelu"),
            layers.Dense(7, activation="gelu"),
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
