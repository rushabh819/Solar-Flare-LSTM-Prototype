from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        at = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        return -tf.reduce_mean(at * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return loss


def build_lstm_classifier(
    sequence_length: int,
    n_features: int,
    hidden_1: int = 64,
    hidden_2: int = 32,
    dropout: float = 0.25,
    learning_rate: float = 1e-3,
    use_focal_loss: bool = True,
):
    inputs = keras.Input(shape=(sequence_length, n_features), name="sequence")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LSTM(hidden_1, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(hidden_2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="flare_prob")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    loss = binary_focal_loss() if use_focal_loss else "binary_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model
