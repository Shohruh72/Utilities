# Facial Landmark Datasets Preprocessing Toolkit

## Description

The Facial Landmark Preprocessing Toolkit is designed to facilitate the preprocessing of facial images and their corresponding landmark annotations. This toolkit includes functions for reading and processing images, extracting and normalizing facial landmarks, and saving the processed data for both training and testing purposes.
## Features
* Crop and resize images to a target size while maintaining the aspect ratio.
* Normalize facial landmark annotations to the cropped images.
* Process multiple facial landmark datasets.
* Save processed images and annotations for training and test sets.
* Calculate and save the mean face landmark configuration.


## Getting Started

### Dependencies

* Python 3.6 or higher
* OpenCV
* NumPy
* Pillow

## Functions

_**Briefly describe each function provided in the script. For instance:_**

* **get_sub_imgs(path, target_path):** Moves images with certain extensions from a given path to a target path, avoiding naming conflicts.
* **collect_files(src1, src2, destination):** Merges files from two source directories into a single destination directory.
* **verify_name_pairs(image_path, label_path):** Checks for naming mismatches between images and corresponding label files.
* **visualize_landmarks(image_name, pts_name, output_name=None):** Visualizes facial landmarks on an image given the image file and the corresponding .pts file.
* **split_dataset(main_folder, train_folder, test_folder, test_ratio=0.2):** Splits the dataset into training and testing sets based on a given ratio.
* **txt2pts(path):** Converts annotations from .txt format to .pts format.
* **rename(path):** Renames .pts files according to a new naming convention.
* **remove_file(path):** Removes files with specific extensions from a directory.
* **copy_files():** Copies files listed in a CSV from one directory to another based on their names.
* **make_names():** Processes image names from a CSV file and saves them to a new CSV file.
 
