import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = r"D:\Mini Project\Code\brain_tumor_dataset"

yes_path = os.path.join(DATA_DIR, 'yes')
no_path = os.path.join(DATA_DIR, 'no')

num_yes_images = len(os.listdir(yes_path))
num_no_images = len(os.listdir(no_path))

initial_image_counts_text = f"Number of Tumor (Positive) Images: {num_yes_images}\nNumber of No Tumor (Negative) Images: {num_no_images}\nTotal Images in Dataset: {num_yes_images + num_no_images}"

plt.figure(figsize=(6, 4))
plt.text(0.01, 0.99, initial_image_counts_text, {'fontsize': 12, 'fontname': 'monospace'}, va='top', ha='left')
plt.axis('off')
plt.title('Initial Dataset Image Counts')
plt.show()

print("\nDisplaying sample images from the dataset:")

yes_dir = os.path.join(DATA_DIR, 'yes')
no_dir = os.path.join(DATA_DIR, 'no')

yes_images = [os.path.join(yes_dir, img) for img in os.listdir(yes_dir)[:3]]
no_images = [os.path.join(no_dir, img) for img in os.listdir(no_dir)[:3]]  

plt.figure(figsize=(10, 5))
for i, img_path in enumerate(yes_images):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Tumor ({i+1})")
    plt.axis('off')

for i, img_path in enumerate(no_images):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    plt.subplot(2, 3, i + 4)
    plt.imshow(img)
    plt.title(f"No Tumor ({i+1})")
    plt.axis('off')
plt.tight_layout()
plt.show()

print("Sample images displayed.")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

def get_filepaths_and_labels(data_dir):
    filepaths = []
    labels = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                filepaths.append(os.path.join(class_path, img_name))
                labels.append(1 if class_name == 'yes' else 0)
    return np.array(filepaths), np.array(labels)

filepaths, labels = get_filepaths_and_labels(DATA_DIR)

from sklearn.model_selection import train_test_split

X_train, X_test_val, y_train, y_test_val = train_test_split(filepaths, labels, test_size=0.2, random_state=42, stratify=labels)

X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val)

image_counts_text = f"Total images: {len(filepaths)}\nTraining images: {len(X_train)}\nValidation images: {len(X_val)}\nTest images: {len(X_test)}"

plt.figure(figsize=(6, 4))
plt.text(0.01, 0.99, image_counts_text, {'fontsize': 12, 'fontname': 'monospace'}, va='top', ha='left')
plt.axis('off')
plt.title('Dataset Split Image Counts')
plt.show()

train_generator = train_datagen.flow_from_dataframe(
    pd.DataFrame({'filepaths': X_train, 'labels': y_train.astype(str)}),
    x_col='filepaths',
    y_col='labels',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=42
)

validation_generator = train_datagen.flow_from_dataframe(
    pd.DataFrame({'filepaths': X_val, 'labels': y_val.astype(str)}),
    x_col='filepaths',
    y_col='labels',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=42
)

test_generator = test_datagen.flow_from_dataframe(
    pd.DataFrame({'filepaths': X_test, 'labels': y_test.astype(str)}),
    x_col='filepaths',
    y_col='labels',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Data preprocessing setup complete. Generators created for training and validation.")

labels = ['Training Images', 'Validation Images', 'Test Images']
counts = [train_generator.samples, validation_generator.samples, test_generator.samples]

plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.ylabel('Number of Images')
plt.title('Image Counts per Data Split (from Generators)')
plt.show()

base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

print("Model architecture built using MobileNetV2 and compiled.")

EPOCHS = 15

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

print("Model training complete.")

from sklearn.metrics import confusion_matrix, classification_report

print("Evaluating model on test data...")

y_true = test_generator.classes

y_pred_proba = model.predict(test_generator)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['no_tumor', 'tumor'], yticklabels=['no_tumor', 'tumor'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Heatmap')
plt.show()

report = classification_report(y_true, y_pred, target_names=['no_tumor', 'tumor'])

plt.figure(figsize=(8, 6))
plt.text(0.01, 0.99, report, {'fontsize': 12, 'fontname': 'monospace'}, va='top', ha='left')
plt.axis('off')
plt.title('Classification Report')
plt.show()

print("\nSample Predictions:")

num_samples_to_show = 5

for i in range(num_samples_to_show):
    print(f"Image: {test_generator.filenames[i]}, True Label: {y_true[i]}, Predicted: {y_pred[i]}")

print("Model evaluation complete.")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

print("Accuracy and Loss graphs generated.")

