
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


TRAIN_DIR = "/content/drive/MyDrive/big_five_dataset_70_30/train"
TEST_DIR  = "/content/drive/MyDrive/big_five_dataset_70_30/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 30


train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=12,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=8,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes
CLASS_NAMES = list(train_data.class_indices.keys())

print("Classes:", CLASS_NAMES)


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)


base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

print("\n🚀 Phase 1 Training Started...")
model.fit(
    train_data,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping(patience=6, restore_best_weights=True)
    ]
)


base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-6),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

print("\n🔧 Fine-Tuning Started...")
model.fit(
    train_data,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping(patience=8, restore_best_weights=True),
        ReduceLROnPlateau(patience=4, factor=0.3, min_lr=1e-7)
    ]
)


test_loss, test_acc = model.evaluate(test_data)
print(f"\n✅ FINAL TEST ACCURACY: {test_acc * 100:.2f}%")

y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data), axis=1)

print("\n📊 CLASSIFICATION REPORT\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


model.save("mobilenetv2_handwriting_personality_optimized.keras")
print("✅ Model saved successfully!")
