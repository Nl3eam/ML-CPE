import json
import os

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing import VOCAB_SIZE

# 1-D VGG layout: (filters, number of conv layers) per block.
VGG_1D = [(32, 2), (64, 2), (128, 3)]

# Lighter variant.
VGG_SMALL = [(32, 2), (64, 2)]


def vgg_block(model, filters, n_conv, kernel_size=3):
    """n_conv x (Conv1D -> BatchNorm) then halve the sequence length."""

    for _ in range(n_conv):
        model.add(layers.Conv1D(filters, kernel_size, padding="same",
                                activation="relu"))
        model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling1D(2))


def build_model(input_shape, num_classes, blocks=VGG_1D, embed_dim=32):

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape, dtype="int32"))

    # Character ids -> dense vectors (replaces Rescaling of pixels)
    model.add(layers.Embedding(VOCAB_SIZE, embed_dim))
    model.add(layers.SpatialDropout1D(0.1))

    for filters, n_conv in blocks:
        vgg_block(model, filters, n_conv)

    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=50, batch_size=32, blocks=VGG_1D):

    model = build_model(X_train.shape[1:], num_classes, blocks)
    model.summary()

    # Classes are imbalanced (Pack >> Tray), so weight rare classes higher
    weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y_train
    )
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print("Class weights:", class_weight)

    callbacks = [
        # val_loss can look "best" for a model that just predicts the majority
        # class, so stop on validation accuracy and wait longer.
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=70, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model.save(os.path.join(output_dir, "vgg_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, 'vgg_model.keras')}")

    return model, history


def predict_model(model, X_test):

    probabilities = model.predict(X_test, verbose=0)

    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)