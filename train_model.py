import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation

path = "./img_with_label_mid/"
# Get list of image files, including subdirectories
image_files = glob.glob(path + "/*.jpg", recursive=True)

# Verify images are found
print("Found image files:", image_files[:5])
print("Total number of images:", len(image_files))

# Step 1: Define a dataset class for efficient loading
class GTAImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = self.image_files[idx]

        # Open image, convert to grayscale, and resize to 224x224
        img = Image.open(file_path)
        img_array = np.array(img) / 255.0  # Normalize

        # Extract label from filename
        filename = file_path.split("/")[-1]
        match = re.search(r"_(\d{4})\.jpg", filename)
        label = [int(char) for char in match.group(1)] if match else [0, 0, 0, 0]

        return img_array[np.newaxis, ...], np.array(label)  # Add channel dim

# Step 2: Create dataset and split into train/test
dataset = GTAImageDataset(image_files)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Step 3: Use a DataLoader for batch processing
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Step 4: Define the model with adjusted input shape (224x224)
# Define CNN architecture
model = keras.Sequential([
    # First block
    Conv2D(32, (5, 5), activation='relu', input_shape=(78, 224, 3), strides=(2, 2)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Second block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Third block
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Fully connected layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),  # Reduce dropout
    Dense(4, activation='sigmoid')  # 4 independent binary labels (WASD)
])

# Compile with AdamW optimizer (better generalization)
from tensorflow.keras.optimizers import AdamW
model.compile(optimizer=AdamW(learning_rate=5e-4, weight_decay=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Step 5: Train the model using the DataLoader
train_images, train_labels = zip(*list(train_loader))
train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)

test_images, test_labels = zip(*list(test_loader))
test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

# Convert PyTorch Tensors from (batch, 1, height, width) -> (batch, height, width, 1) for TensorFlow
def preprocess_batch(data_loader):
    images, labels = zip(*list(data_loader))
    labels = np.concatenate(labels)
    # images = np.concatenate(images).transpose(0, 2, 3, 1)  # Move channel to last
    images = np.concatenate(images)
    images = np.squeeze(images, axis=1)  # Move channel to last
    return images, labels

# Prepare training and testing sets
train_images, train_labels = preprocess_batch(train_loader)
test_images, test_labels = preprocess_batch(test_loader)

# Confirm the new shape (batch, height, width, channels)
print("Train images shape:", train_images.shape)  # (batch, 224, 224, 1)
print("Test images shape:", test_images.shape)

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Save the model
model.save('GTA_model.keras')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()