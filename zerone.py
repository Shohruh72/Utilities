# Importing required libraries
import os
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET

file_path = '../Datasets/dataserver/IR/zer01ne/face_annotations.xml'
output_dir = '../Datasets/dataserver/IR/zer01ne/xml/'
output_dir2 = '../Datasets/dataserver/IR/zer01ne/pts/'
tree = ET.parse(file_path)
root = tree.getroot()

index_ranges = {
    'jaw': (1, 17),
    'right_eyebrow': (18, 22),
    'left_eyebrow': (23, 27),
    'nose': (28, 36),
    'right_eye': (37, 42),
    'left_eye': (43, 48),
    'mouth': (49, 60),
    'inner_mouth': (61, 68)
}


def xml_to_pts(xml_filepath, pts_filepath):
    # Parse the XML file
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    # Extract facial landmark points
    points = []
    for point_elem in root.findall('.//points'):
        if 'points' in point_elem.attrib:
            coords = point_elem.attrib['points'].split(',')
            if len(coords) == 2:
                points.append([float(coords[0]), float(coords[1])])

    # Check if points were found
    if not points:
        print(f"No points found in {os.path.basename(xml_filepath)}")
        return

    # Convert points to numpy array for easier handling
    points = np.array(points)

    # Write to PTS file
    with open(pts_filepath, 'w') as f:
        f.write("version: 1\n")
        f.write(f"n_points:  {len(points)}\n")
        f.write("{\n")
        for point in points:
            f.write(f"{point[0]} {point[1]}\n")
        f.write("}\n")

    print(f"Successfully converted: {os.path.basename(xml_filepath)} -> {os.path.basename(pts_filepath)}")


def save_individual_xml_corrected(image_elem, output_dir, index_ranges):
    # Creating a new XML tree
    new_tree = ET.ElementTree(ET.Element('annotations'))
    new_root = new_tree.getroot()

    # Adding the image element to the new XML tree
    new_root.append(image_elem)

    # Getting the image ID to create a filename
    image_id = image_elem.get('name').split('.')[0]
    filename = f"{image_id}.xml"
    filepath = os.path.join(output_dir, filename)

    # Correcting the indexing of facial landmarks
    for skeleton_elem in image_elem.findall('.//skeleton'):
        label = skeleton_elem.get('label')
        if label in index_ranges:
            index_range = index_ranges[label]
            for i, point_elem in enumerate(skeleton_elem.findall('.//points'), start=index_range[0]):
                point_elem.set('label', str(i))

    # Saving the new XML tree to a file
    new_tree.write(filepath, encoding='utf8')


for image_elem in root.findall('.//image'):
    save_individual_xml_corrected(image_elem, output_dir, index_ranges)

output_dir


def xml_to_pts_based_on_indexes(xml_filepath, pts_filepath):
    # Parse the XML file
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    # Extract facial landmark points with their indexes
    points_dict = {}
    for point_elem in root.findall('.//points'):
        if 'points' in point_elem.attrib and 'label' in point_elem.attrib:
            index = int(point_elem.attrib['label'])
            coords = point_elem.attrib['points'].split(',')
            if len(coords) == 2:
                points_dict[index] = [float(coords[0]), float(coords[1])]

    # Check if points were found
    if not points_dict:
        print(f"No points found in {os.path.basename(xml_filepath)}")
        return

    # Sort the points based on their indexes
    sorted_points = [points_dict[i] for i in sorted(points_dict.keys())]

    # Convert points to numpy array for easier handling
    points = np.array(sorted_points)

    # Write to PTS file
    with open(pts_filepath, 'w') as f:
        f.write("version: 1\n")
        f.write(f"n_points:  {len(points)}\n")
        f.write("{\n")
        for point in points:
            f.write(f"{point[0]} {point[1]}\n")
        f.write("}\n")

    print(f"Successfully converted: {os.path.basename(xml_filepath)} -> {os.path.basename(pts_filepath)}")


def convert_xml_to_pts_based_on_indexes(input_dir, output_dir):
    # Find all XML files in the specified directory
    xml_files = glob.glob(os.path.join(input_dir, '*.xml'))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each XML file
    for xml_file in xml_files:
        # Creating the PTS filename based on the XML filename
        pts_filename = os.path.basename(xml_file).replace('.xml', '.pts')
        pts_filepath = os.path.join(output_dir, pts_filename)

        # Convert the XML file to PTS format
        xml_to_pts_based_on_indexes(xml_file, pts_filepath)


