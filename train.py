from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "real_and_fake_face"
MODEL_OUT = ROOT / "models" / "deepfake_detection_model.h5"
ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def build_dummy_model():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, InputLayer

    model = Sequential([
        InputLayer(input_shape=(96, 96, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_real_model():
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=(96, 96),
        batch_size=32,
        class_mode="sparse",
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=(96, 96),
        batch_size=32,
        class_mode="sparse",
        subset="validation",
        shuffle=False
    )

    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
    base.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.1),
        Dense(2, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def scheduler(epoch):
        if epoch <= 2:
            return 1e-3
        elif epoch <= 15:
            return 1e-4
        else:
            return 1e-5

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    hist = model.fit(
        train_gen,
        epochs=5,  # kör kort först
        validation_data=val_gen,
        callbacks=[lr_callback],
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
    )

    # Spara enkla plots (optional)
    try:
        loss_fig = ASSETS_DIR / "Figure_1.png"
        acc_fig = ASSETS_DIR / "Figure_2.png"

        plt.figure(figsize=(7, 5))
        plt.plot(hist.history["loss"])
        plt.plot(hist.history["val_loss"])
        plt.title("Loss")
        plt.legend(["train", "val"])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_fig)

        plt.figure(figsize=(7, 5))
        plt.plot(hist.history["accuracy"])
        plt.plot(hist.history["val_accuracy"])
        plt.title("Accuracy")
        plt.legend(["train", "val"])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(acc_fig)
    except Exception as e:
        print("Plot save skipped:", e)

    return model

if __name__ == "__main__":
    print(">>> train.py started")
    print("Dataset expected at:", DATASET_DIR)

    if not DATASET_DIR.exists():
        print(">>> No dataset found. Creating a DUMMY model so the app can run end-to-end.")
        model = build_dummy_model()
    else:
        print(">>> Dataset found. Training real model.")
        model = train_real_model()

    model.save(str(MODEL_OUT))
    print(">>> Model saved to:", MODEL_OUT)
