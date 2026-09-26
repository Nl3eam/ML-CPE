import json
import os

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_length, vocab_size, num_classes, embedding_dim=64):
    """1D convolutional network: 3 conv blocks -> dense head.

    Same shape as the image CNN, one dimension down: an Embedding layer
    replaces Rescaling, Conv1D replaces Conv2D, and the filters slide over
    word positions instead of pixels.
    """

    model = keras.Sequential([
        keras.Input(shape=(input_length,), dtype="int32"),

        # Turn word ids into dense vectors inside the model, so inference
        # code only has to hand over integer sequences
        layers.Embedding(vocab_size, embedding_dim, mask_zero=False),

        # Dropping whole embedding channels is the text equivalent of the
        # random flips/zooms used on images: it stops a small net from
        # memorising single words
        layers.SpatialDropout1D(0.2),

        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        # Max over positions: keeps the strongest n-gram signal wherever
        # in the review it appeared
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        # 1 sigmoid output for 2 classes, softmax otherwise
        layers.Dense(
            1 if num_classes == 2 else num_classes,
            activation="sigmoid" if num_classes == 2 else "softmax"
        ),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val, y_val, vocab_size, num_classes,
                output_dir=None, epochs=30, batch_size=32):
    """Build, train and save the model. Returns (model, history)."""

    model = build_model(X_train.shape[1], vocab_size, num_classes)
    model.summary()

    callbacks = [
        # Stop when validation loss stops improving, keep the best weights
        #keras.callbacks.EarlyStopping(
         #   monitor="val_loss", patience=10 , restore_best_weights=True
        #),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model.save(os.path.join(output_dir, "cnn_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, 'cnn_model.keras')}")

    return model, history


def predict_model(model, X_test):

    probabilities = model.predict(X_test, verbose=0)

    # Binary head outputs one probability, multiclass outputs one per class
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)