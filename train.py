from pathlib import Path

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

dataset_path = Path(__file__).resolve().parent / "real_and_fake_face_detection" / "real_and_fake_face"
class_names = ("training_fake", "training_real")
missing_classes = [name for name in class_names if not (dataset_path / name).is_dir()]
if missing_classes:
    raise FileNotFoundError(
        f"Dataset folders not found under {dataset_path}: {', '.join(missing_classes)}"
    )

train_data = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    validation_split=0.2,
)
validation_data = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)
train = train_data.flow_from_directory(
    str(dataset_path),
    class_mode="sparse",
    target_size=(96, 96),
    batch_size=32,
    subset="training",
)
val = validation_data.flow_from_directory(
    str(dataset_path),
    class_mode="sparse",
    target_size=(96, 96),
    batch_size=32,
    subset="validation",
    shuffle=False,
)

# MobileNetV2 model
tf.keras.backend.clear_session()
mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))

model = Sequential([mnet,
                    GlobalAveragePooling2D(),
                    Dense(512, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(128, activation="relu"),
                    Dropout(0.1),
                    Dense(2, activation="softmax")])

model.layers[0].trainable = False

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    ModelCheckpoint(
        "deepfake_detection_model.h5",
        monitor="val_loss",
        save_best_only=True,
    ),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
]
history = model.fit(train, epochs=20, callbacks=callbacks, validation_data=val)

# Save model
model.save('deepfake_detection_model.h5')

# Visualizing accuracy and loss
epochs = len(history.history["loss"])
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
xc = range(1, epochs + 1)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True)
plt.legend(['Train', 'Validation'])
plt.style.use(['classic'])
plt.savefig("Figure_1.png", bbox_inches="tight")

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(['Train', 'Validation'], loc=4)
plt.style.use(['classic'])
plt.savefig("Figure_2.png", bbox_inches="tight")
plt.show()
