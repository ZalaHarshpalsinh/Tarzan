import cv2
import numpy as np
import os
import time
from datetime import datetime
from pynput.keyboard import Controller
from mss import mss  # For screen capture
import tensorflow as tf

# Create a keyboard controller
keyboard = Controller()

# Load the trained model
MODEL_PATH = "./GTA_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Folder to save captured frames
SAVE_FOLDER = "./content/captured_frames/"
os.makedirs(SAVE_FOLDER, exist_ok=True)  # Create folder if it doesn't exist

# Define the screen capture region (adjust based on your game window)
SCREEN_REGION = {"top": 640-224, "left": 480-78, "width": 224, "height": 78}

# Target image size
RESIZED_WIDTH = 224
RESIZED_HEIGHT = 78

# Function to capture the screen
def capture_frame():
    with mss() as sct:
        screenshot = sct.grab(SCREEN_REGION)
        img = np.array(screenshot)  # Convert to NumPy array
        
        # Generate a timestamped filename
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Example: "20250221_153045_123456"
        # filename = os.path.join(SAVE_FOLDER, f"frame_{timestamp}.png")

        # # Save the image
        # cv2.imwrite(filename, img)
        # print(f"Saved: {filename}")

        return img

# Function to preprocess image before feeding it to the model
def preprocess_frame(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color format

        # Resize image to 224x224
        frame = cv2.resize(frame, (RESIZED_WIDTH, RESIZED_HEIGHT))

        # Normalize pixel values (0-255 → 0-1)
        frame = frame / 255.0

        # Reshape to add channel dimension (224, 224) → (224, 224, 1)
        # frame = np.expand_dims(frame, axis=-1)  

        # # Add batch dimension (1, 224, 224, 1) for model input
        frame = np.expand_dims(frame, axis=0)

        return frame

# Dummy AI function that decides which key to press
def get_predicted_keys(frame):
        """
        This function should be replaced with a real AI model.
        For now, it randomly chooses a key from 'WASD'.
        """
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)  
        predictions = model.predict(processed_frame)  # Get predictions (4 outputs)

        # Convert predictions into binary keypresses (threshold = 0.5)
        key_flags = (predictions > 0.5).astype(int).flatten()  # Example: [1, 0, 1, 0]
        # Map key_flags to actual keys
        keys = ['w', 'a', 's', 'd']
        keys_to_press = [keys[i] for i in range(4) if key_flags[i] == 1]

        return keys_to_press  # Returns a list of keys to press

# Main loop
if __name__ == "__main__":
        print("Starting game automation...")
        # time.sleep(5)
        try:
                while True:
                        # Capture the game frame
                        frame = capture_frame()

                        # Get AI model predictions
                        keys_to_press = get_predicted_keys(frame)
                        print(f"Pressing keys: {keys_to_press}")

                        # Simulate keypress
                        for key in keys_to_press:
                                keyboard.press(key)
                        
                        time.sleep(1)  # Hold keys for 500ms
                        
                        for key in keys_to_press:
                                keyboard.release(key)

                        # Wait 500ms before next capture
                        time.sleep(1)

        except KeyboardInterrupt:
                print("Stopped by user.")
