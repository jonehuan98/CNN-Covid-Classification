Requirements: Python 3.0 or later
Libraries used: numpy, cv2, torch, torch, matplotlib, tqdm

To run this code, user must have files covid19_dataset_32_32.zip and covid19_dataset_800_800.zip on their local machine.

The code is written to train a Convolutional Neural Network to classify Covid-19 related images.
More specifically, the dataset consists three types of CT Scan images (a) Covid; (b) Normal and (c) Viral Pneumonia.

Each file in the dataset (covid19 dataset 32 32.zip) is a grayscale
image of CT Scan either belonging a Covid-19, Viral Pneumonia or a normal patient. Each iamge
is reshaped to size R33232.

Each file in covid19 dataset 800 800.zip folder is a grayscale
image of CT Scan belonging to either Covid-19, Viral Pneumonia or a normal patient. Each iamge
is reshaped to size R3800800.