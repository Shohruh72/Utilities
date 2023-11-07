import os
import random
import shutil
from collections import Counter

import cv2
import pandas as pd


def get_sub_imgs(path, target_path):
    extensions = ['.pts', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

    os.makedirs(target_path, exist_ok=True)
    for dirs, _, files in os.walk(path):
        if dirs != path:
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in extensions):
                    source = os.path.join(dirs, filename)

                    target = os.path.join(target_path, filename)
                    base, ext = os.path.splitext(target)
                    counter = 1
                    while os.path.exists(target):
                        target = f"{base}_{counter}{ext}"
                        counter += 1

                    shutil.move(source, target)


def collect_files(src1, src2, destination):
    os.makedirs(destination, exist_ok=True)

    def copy_from_source(src):
        for filename in os.listdir(src):
            file_path = os.path.join(src, filename)
            if os.path.isfile(file_path):
                shutil.move(file_path, os.path.join(destination, filename))

    copy_from_source(src1)
    copy_from_source(src2)


def verify_name_pairs(image_path, label_path):
    # List directory contents
    images = os.listdir(image_path)
    labels = os.listdir(label_path)

    # Extract filenames without extensions and filter based on desired extensions
    image_names = [os.path.splitext(im)[0] for im in images if im.endswith(('.png'))]
    label_names = [os.path.splitext(lab)[0] for lab in labels if lab.endswith('.pts')]

    # Find unmatched image and label names
    diff_img_label = list((Counter(image_names) - Counter(label_names)).elements())
    diff_label_img = list((Counter(label_names) - Counter(image_names)).elements())

    # Report any mismatches
    if diff_img_label or diff_label_img or len(images) != len(labels):
        print("error comes from images folder:", diff_img_label)
        print("error comes from label folder:", diff_label_img)
        print('Please check folders.')
    else:
        print('OK...')


def visualize_landmarks(image_name, pts_name, output_name=None):
    # Read landmark points from .pts file
    with open(pts_name, 'r') as file:
        content = file.readlines()

    # Extracting the lines containing landmark points
    landmarks_data = [line.strip() for line in content if " " in line]

    # Converting the string points to tuples of floats
    landmarks = [tuple(map(float, line.split())) for line in landmarks_data[3:-1]]

    # Reading the image
    img = cv2.imread(image_name)

    # Check if the image is loaded successfully
    if img is None:
        print("Error: Unable to load image.")
        return

    # Drawing the facial landmarks on the image
    for idx, (x, y) in enumerate(landmarks):
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)  # Red circle
        # cv2.putText(img, str(idx + 1), (int(x) + 2, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255),
        #             1)  # White text

    # Displaying the image
    cv2.imshow("Facial Landmarks", img)

    # Saving the image if output_name is provided
    if output_name:
        cv2.imwrite(output_name, img)

    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def split_dataset(main_folder, train_folder, test_folder, test_ratio=0.2):
    image_extensions = ['.png']
    label_extensions = ['.pts']
    # Ensure the train and test folders exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all image files from the main folder
    image_files = [f for f in os.listdir(main_folder) if os.path.isfile(os.path.join(main_folder, f)) and any(
        f.lower().endswith(ext) for ext in image_extensions)]

    # Shuffle and split the files
    random.shuffle(image_files)
    split_index = int(len(image_files) * (1 - test_ratio))
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    # Move files function (nested within the main function)
    def move_files(files, source_folder, destination_folder):
        for filename in files:
            # Move the image file
            shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folder, filename))

            # Move the corresponding label file
            base_name, _ = os.path.splitext(filename)
            for ext in label_extensions:
                label_file = f"{base_name}{ext}"
                label_path = os.path.join(source_folder, label_file)
                if os.path.exists(label_path):
                    shutil.move(label_path, os.path.join(destination_folder, label_file))

    # Call the move files function for train and test data
    move_files(train_files, main_folder, train_folder)
    move_files(test_files, main_folder, test_folder)


def txt2pts(path):
    for f_name in os.listdir(path):
        if f_name.endswith('.txt'):
            with open(path + '/' + f_name, 'r') as file:
                all_lines = file.readlines()

            num_lines = len(all_lines)
            if num_lines < 70:
                raise ValueError("The file contains less than 70 lines. Cannot remove the 69th and 70th lines.")
            modified_lines = all_lines[:68] + all_lines[70:]
            new_pts_name = f'{path}/{f_name[:-4]}' + '.pts'
            with open(new_pts_name, 'w') as file:
                file.writelines(modified_lines)


def rename(path):
    for f_name in os.listdir(path):
        if f_name.endswith('.pts'):
            print(f_name)
            with open(path + f_name, 'r') as f:
                contents = f.read()
            os.rename((path + f_name), (f'{path}{f_name[:-10]}' + '.pts'))
            new_name = f'{f_name[:-10]}' + '.pts'
            with open(new_name, 'w') as f:
                f.writelines(contents)


def remove_file(path):
    for img_name in os.listdir(path):
        if img_name.endswith('.txt') or img_name.endswith('_seg.png'):
            full_path = os.path.join(path + img_name)
            print(full_path)
            os.remove(full_path)


def copy_files():
    csv_path = ''
    destination_directory = ''
    root_directory = ''
    image_names = pd.read_csv(csv_path, header=None)[0].tolist()
    # Create a set for faster lookup
    image_names_set = set(image_names)
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Search for images and copy them to the destination directory
    i = 0
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file[:-4] in image_names_set:
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_directory, file)

                # Copy image to destination directory
                shutil.move(source_path, destination_path)
                i += 1
                print(f"Image {file} copied to {destination_directory}")
    print('number: ', i)
    print("Image copying process is complete.")


def make_names():
    input_csv_path = ''

    output_csv_path = ''

    df = pd.read_csv(input_csv_path)

    image_names = df.iloc[:, 0]

    image_names = image_names.apply(lambda x: x.split("/")[-1][:-4])
    print()
    image_names.to_csv(output_csv_path, index=False, header=False)

    image_names.head(), output_csv_path
