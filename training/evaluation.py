"""
EVALUATION SCRIPT FOR ALREADY TRAINED MODEL
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------- PATHS -------------
MODEL_PATH = r"C:\Users\hp\Handwriting_Traits\backend\models\personality_efficientnet.h5"
TEST_DATA_PATH = r"C:\Users\hp\Handwriting_Traits\dataset\dataset_split\dataset_split\test"

# ----------- LOAD MODEL ----------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ----------- LOAD TEST DATA ----------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

test_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255).flow_from_directory(
    TEST_DATA_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_labels = list(test_data.class_indices.keys())
print("\nClass Labels:", class_labels)

# ----------- PREDICT --------------
print("\nPredicting on test dataset...")
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

# True labels
y_true_classes = test_data.classes

# ----------- ACCURACY ----------
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print("\n======================================")
print(" Test Accuracy:", accuracy)
print("======================================\n")

# ----------- CLASSIFICATION REPORT ----------
report = classification_report(y_true_classes, y_pred_classes, target_names=class_labels)
print("Classification Report:\n")
print(report)

# Save report to file
with open("evaluation_results.txt", "w") as f:
    f.write("Accuracy: " + str(accuracy) + "\n\n")
    f.write(report)

print("\nEvaluation results saved to evaluation_results.txt")

# ----------- CONFUSION MATRIX ----------
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
