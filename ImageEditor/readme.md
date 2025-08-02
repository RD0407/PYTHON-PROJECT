Image Sharpening Batch Script

A simple Python project to batch-sharpen all images in a folder and save the results to a new directory.
Useful for quickly improving image clarity for many files at once.
Features

    Batch Processing: Automatically processes all images in a chosen folder.

    Image Sharpening: Applies a sharpening filter to each image to enhance clarity.

    Easy to Use: Just place your images in the imgs/ folder and run the script.

    Safe Output: Saves all results into a new folder called editedImgs/ to keep originals untouched.

How to Use

    Put your images (e.g. .png, .jpg) into the imgs/ folder (create it if it doesn’t exist).

    Make sure you have the required packages installed:

pip install pillow

Run the script:

    python script_name.py

    Check your edited images in the editedImgs/ folder.
    All edited files will have _edited appended to the original name.

Sample Code

import os
from PIL import Image, ImageEnhance, ImageFilter

path = 'imgs'
pathOut = 'editedImgs'

os.makedirs(pathOut, exist_ok=True)

for filename in os.listdir(path):
    img = Image.open(os.path.join(path, filename))
    edit = img.filter(ImageFilter.SHARPEN)
    clean_name = os.path.splitext(filename)[0]
    edit.save(os.path.join(pathOut, f'{clean_name}_edited.png'))

Folder Structure

project/
├── imgs/            # Input images here
├── editedImgs/      # Output folder for edited images
├── script_name.py   # This script
└── README.md        # (this file)

Requirements

    Python 3.x

    Pillow (pip install pillow)

License

Feel free to use or adapt this script for any project!

Let me know if you want to add more details (like customization, more filters, or example images)!
