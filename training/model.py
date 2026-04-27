"""Notebook-aligned TensorFlow model definition."""

import tensorflow as tf
from tensorflow.keras import Model, layers

from config import FILTERS, META_DIM, NUM_SE_BLOCKS


def se_block(x, channels: int, ratio: int = 4):
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(channels // ratio, 8), activation="relu", use_bias=False)(s)
    s = layers.Dense(channels, activation="sigmoid", use_bias=False)(s)
    s = layers.Reshape((1, 1, channels))(s)
    return layers.Multiply()([x, s])


def res_se_block(x, filters: int):
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = se_block(y, filters)
    y = layers.Add()([x, y])
    return layers.Activation("relu")(y)


def build_model() -> Model:
    board_in = layers.Input(shape=(8, 8, 12), name="board")
    meta_in = layers.Input(shape=(META_DIM,), name="meta")

    x = layers.Conv2D(FILTERS, 3, padding="same", use_bias=False, name="stem")(board_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for _ in range(NUM_SE_BLOCKS):
        x = res_se_block(x, FILTERS)

    x = layers.Flatten()(x)
    # Keep this cast layer fully serializable across Keras versions.
    x = layers.Activation("linear", dtype="float32", name="board_flat_f32")(x)
    z = layers.Concatenate()([x, meta_in])
    z = layers.Dense(512, activation="relu", dtype="float32")(z)
    z = layers.Dense(256, activation="relu", dtype="float32")(z)
    out = layers.Dense(1, activation="linear", name="eval", dtype="float32")(z)

    return Model(inputs=[board_in, meta_in], outputs=out, name="chess_se_resnet_meta")
