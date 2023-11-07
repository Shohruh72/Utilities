import os

import cv2
import numpy


def process(data_dir, folder, image_name, label_name, target_size):
    image_path = os.path.join(data_dir, folder, image_name)
    label_path = os.path.join(data_dir, folder, label_name)

    with open(label_path, 'r') as f:
        annotation = f.readlines()
        annotation = [x.strip().split() for x in annotation]
        annotation = [[int(float(x[0])), int(float(x[1]))] for x in annotation]

        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        anno_x = [x[0] for x in annotation]
        anno_y = [x[1] for x in annotation]
        x_min = min(anno_x)
        y_min = min(anno_y)
        x_max = max(anno_x)
        y_max = max(anno_y)
        box_w = x_max - x_min
        box_h = y_max - y_min
        scale = 1.1
        x_min -= int((scale - 1) / 2 * box_w)
        y_min -= int((scale - 1) / 2 * box_h)
        box_w *= scale
        box_h *= scale
        box_w = int(box_w)
        box_h = int(box_h)
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        box_w = min(box_w, image_width - x_min - 1)
        box_h = min(box_h, image_height - y_min - 1)
        annotation = [[(x - x_min) / box_w, (y - y_min) / box_h] for x, y in annotation]

        x_max = x_min + box_w
        y_max = y_min + box_h
        image_crop = image[y_min:y_max, x_min:x_max, :]
        image_crop = cv2.resize(image_crop, (target_size, target_size))
        return image_crop, annotation


def convert(data_dir, target_size=256):
    if not os.path.exists(os.path.join(data_dir, 'images', 'train')):
        os.makedirs(os.path.join(data_dir, 'images', 'train'))
    if not os.path.exists(os.path.join(data_dir, 'images', 'test')):
        os.makedirs(os.path.join(data_dir, 'images', 'test'))

    folders = ['train/300W', 'train/aihub_dms', 'train/COFW', 'train/dibox', 'train/frgc', 'train/Menpo2D',
               'train/MultiPIE', 'train/nir_face', 'train/synthetic', 'train/xm2vts']
    annotations = {}
    for folder in folders:
        filenames = sorted(os.listdir(os.path.join(data_dir, folder)))
        label_files = [x for x in filenames if '.pts' in x]
        image_files = [x for x in filenames if '.pts' not in x]
        print(f'{folder}:{len(label_files)} ======{len(image_files)}')
        assert len(image_files) == len(label_files)
        for image_name, label_name in zip(image_files, label_files):
            image_crop_name = folder.replace('/', '_') + '_' + image_name
            image_crop_name = os.path.join(data_dir, 'images', 'train', image_crop_name)

            image_crop, annotation = process(data_dir, folder, image_name, label_name, target_size)
            cv2.imwrite(image_crop_name, image_crop)
            annotations[image_crop_name] = annotation
    with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
        for image_crop_name, annotation in annotations.items():
            f.write(image_crop_name + ' ')
            for x, y in annotation:
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')

    annotations = {}
    folders = ['test/300W', 'test/aihub_dms', 'test/COFW', 'test/dibox', 'test/frgc', 'test/Menpo2D', 'test/MultiPIE',
               'test/nir_face', 'test/xm2vts']
    folders = ['test']
    for folder in folders:
        filenames = sorted(os.listdir(os.path.join(data_dir, folder)))
        label_files = [x for x in filenames if '.pts' in x]
        image_files = [x for x in filenames if '.pts' not in x]
        assert len(image_files) == len(label_files)
        for image_name, label_name in zip(image_files, label_files):
            image_crop_name = folder.replace('/', '_') + '_' + image_name
            image_crop_name = os.path.join(data_dir, 'images', 'test', image_crop_name)

            image_crop, annotation = process(data_dir, folder, image_name, label_name, target_size)
            cv2.imwrite(image_crop_name, image_crop)
            annotations[image_crop_name] = annotation
    with open(os.path.join(data_dir, 'test.txt'), 'w') as f:
        for image_crop_name, annotation in annotations.items():
            f.write(image_crop_name + ' ')
            for x, y in annotation:
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')

    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        annotations = f.readlines()
    with open(os.path.join(data_dir, 'test_common.txt'), 'w') as f:
        for annotation in annotations:
            if 'ibug' not in annotation:
                f.write(annotation)
    with open(os.path.join(data_dir, 'test_challenge.txt'), 'w') as f:
        for annotation in annotations:
            if 'ibug' in annotation:
                f.write(annotation)

    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        annotations = f.readlines()
    annotations = [x.strip().split()[1:] for x in annotations]
    annotations = [[float(x) for x in anno] for anno in annotations]
    annotations = numpy.array(annotations)
    mean_face = [str(x) for x in numpy.mean(annotations, axis=0).tolist()]

    with open(os.path.join(data_dir, 'indices.txt'), 'w') as f:
        f.write(' '.join(mean_face))
