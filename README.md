# Project Tarzan :- Autonomous Driving System for GTA Vice City  

This project aims to develop an autonomous driving system for a car in GTA Vice City using a CNN-based supervised learning approach. By capturing frequent screenshots and predicting driving actions, the model navigates the car autonomously within the game environment.

---

## Table of Contents
- [Project Tarzan :- Autonomous Driving System for GTA Vice City](#project-tarzan---autonomous-driving-system-for-gta-vice-city)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Problem Statement](#problem-statement)
  - [Solution Approach](#solution-approach)
  - [Requirements](#requirements)
  - [Usage](#usage)
  - [Results](#results)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

---

## Overview
The project focuses on building an autonomous driving agent for GTA Vice City using deep learning. By leveraging a Convolutional Neural Network (CNN), the system predicts driving actions (e.g., pressing 'W', 'A', 'S', or 'D' keys) based on real-time game screenshots.

---

## Problem Statement
Manual driving in video games can be repetitive and monotonous. Automating this process not only enhances gameplay but also serves as an interesting use case for Computer Vision and Deep Learning applications in autonomous driving.

---

## Solution Approach
1. **Data Collection**:  
    - Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/eryash15/gta-vice-city-car-driving-images-with-label) using `kagglehub`.
    - Consists of images labeled with driving actions extracted from filename patterns.
    - Each image is labeled with a 4-digit vector representing the state of 'W', 'A', 'S', and 'D' keys.
   - Screenshots were resized to 224x224 pixels for consistent input to the CNN model.

2. **Model Architecture**:  
   - A Convolutional Neural Network (CNN) with three layers was used:
     - Convolutional Layer (32 filters) → MaxPooling
     - Convolutional Layer (64 filters) → MaxPooling
     - Convolutional Layer (128 filters) → MaxPooling
     - Fully Connected Layer → Output Layer (4 units for 'W', 'A', 'S', 'D')
   - Activation Function: ReLU for hidden layers, Sigmoid for output.

3. **Training and Testing**:  
   - Dataset split: 80% for training, 20% for testing.
   - Loss Function: Binary Cross-Entropy
   - Optimizer: Adam
   - Evaluation Metrics: Accuracy

---

## Requirements
To set up and run this project, ensure the following dependencies are installed:

```bash
pip install numpy opencv-python-headless Pillow matplotlib torch tensorflow mss pynput kagglehub
```

---

## Usage

---

## Results

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- Kaggle dataset: [GTA Vice City Car Driving Images](https://www.kaggle.com/datasets/eryash15/gta-vice-city-car-driving-images-with-label)

---