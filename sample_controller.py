from pynput.keyboard import Controller
import time

# Create a keyboard controller
keyboard = Controller()

# Function to simulate keypress
def press_key(key, duration=0.1):
        """
        Simulates a key press for a given duration.
        """
        keyboard.press(key)
        time.sleep(duration)
        keyboard.release(key)

# Example: Simulate "WASD" key presses
if __name__ == "__main__":
        print("Simulating keypresses...")

        time.sleep(5)
        press_key('w', 3)  # Move forward
        # press_key('a', )  # Turn left
        press_key('s', 3)  # Move backward
        # press_key('d', 5)  # Turn right

        print("Keypress simulation complete!")
