import os
import shutil

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from io import BytesIO
import numpy as np
import scipy.io as sio
from PIL import Image
from face_detection import RetinaFace

from nets import nn
from utils.util import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data_dir = '../datasets/AFLW2K/'


def ground_truth(file):
    # for file in os.listdir(data_dir):
    #     if file.endswith('.jpg'):
    mat_path = os.path.join(file.replace('.jpg', '.mat'))
    data = sio.loadmat(mat_path)
    pose_para = data['Pose_Para'][0]
    # Extract pitch, yaw, and roll values
    pose_params = pose_para[:3]
    # Convert radians to degrees
    pitch = pose_params[0] * 180 / np.pi
    yaw = pose_params[1] * 180 / np.pi
    roll = pose_params[2] * 180 / np.pi
    return pitch, yaw, roll


def parse_pose_para(file_path='../datasets/example/3545_image00497.mat'):
    data = sio.loadmat(file_path)
    pose_para = data['Pose_Para'][0]
    # Extract pitch, yaw, and roll values
    pose_params = pose_para[:3]
    # Convert radians to degrees
    pitch_deg = pose_params[0] * 180 / np.pi
    yaw_deg = pose_params[1] * 180 / np.pi
    roll_deg = pose_params[2] * 180 / np.pi
    print(pitch_deg, yaw_deg, roll_deg)


def copy_files(data_dir):
    fold = {1: [-45, -35], 2: [-35, -25], 3: [-25, -15], 4: [-15, -5], 5: [-5, 5], 6: [35, 45], 7: [25, 35],
            8: [15, 25], 9: [5, 15]}

    for i in fold.keys():
        a, b = fold[i][0], fold[i][1]
        print(f'{i}: {a}, {b}')

        dist_dir = f'../datasets/roll/{a}{b}'
        os.makedirs(dist_dir, exist_ok=True)

        for file in os.listdir(data_dir):
            if file.endswith('.jpg'):
                mat_path = os.path.join(data_dir, file.replace('.jpg', '.mat'))
                data = sio.loadmat(mat_path)
                pose_para = data['Pose_Para']
                p, y, r = pose_para[0, :3]
                roll = r * 180 / np.pi
                if a < roll <= b:
                    shutil.copy(os.path.join(data_dir, file), dist_dir)
                    shutil.copy(mat_path, dist_dir)
                    print(f"Copied {file} and its .mat file to {dist_dir} due to pitch {roll} degrees")


def demo(file):
    model = torch.load(f=f'outputs/weights/best.pt', map_location='cuda')['model'].float()
    model = nn.re_parameterize_model(model)
    model.cpu().eval()
    model.inference_mode = True

    detector = RetinaFace(0)
    pitch, yaw, roll = None, None, None
    with torch.no_grad():
        frame = cv2.imread(os.path.join(data_dir, file))
        faces = detector(frame)

        for box, landmarks, score in faces:
            if score < .75:
                continue
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)

            x_min = max(0, x_min - int(0.2 * bbox_height))
            y_min = max(0, y_min - int(0.2 * bbox_width))
            x_max = x_max + int(0.2 * bbox_height)
            y_max = y_max + int(0.2 * bbox_width)

            img = frame[y_min:y_max, x_min:x_max]
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = get_transforms(False)(img)

            img = torch.Tensor(img[None, :]).cpu()

            c = cv2.waitKey(1)
            if c == 27:
                break

            R_pred = model(img)

            euler = compute_euler(
                R_pred) * 180 / np.pi
            pitch = euler[:, 0].cpu()
            yaw = euler[:, 1].cpu()
            roll = euler[:, 2].cpu()
            # plot_pose_cube(frame, yaw, pitch, roll, x_min + int(.5 * (
            #         x_max - x_min)), y_min + int(.5 * (y_max - y_min)), size=bbox_width)
            #
            # cv2.imshow("Demo", frame)
            # cv2.waitKey(0)
            # # print(f'pitch: {p_pred_deg.item()}, yaw: {y_pred_deg.item()}, roll: {r_pred_deg.item()}')
            return pitch, yaw, roll
    return pitch, yaw, roll


def categorize_pitch(roll):
    fold = {'-45:-35': [-45, -35], '-35:-25': [-35, -25], '-25:-15': [-25, -15], '-15:-5': [-15, -5], '-5:5': [-5, 5],
            '35:45': [35, 45], '25:35': [25, 35], '15:25': [15, 25], '5:15': [5, 15]}
    for i, (a, b) in fold.items():
        if a < roll <= b:
            return i
    return None


def evaluate(data_dir):
    gt_categories = []
    pred_categories = []

    for file in os.listdir(data_dir):
        if file.endswith('.jpg'):
            gt_pitch, yaw, roll = ground_truth(data_dir + file)
            gt_category = categorize_pitch(yaw)

            # Ensure gt_pitch leads to a valid category
            if gt_category is None:
                continue

            pred_pitch, pred_yaw, pred_roll = demo(file)
            if pred_pitch is not None:
                pred_category = categorize_pitch(pred_yaw.item())
            else:
                pred_category = 'unknown'

            if gt_category is not None:
                gt_categories.append(gt_category)
            else:
                gt_categories.append('Unknown')
            # gt_categories.append(gt_category)
            pred_categories.append(pred_category)

    return gt_categories, pred_categories


def plot_confusion_matrix(gt_categories, pred_categories):
    # Compute confusion matrix
    cf_matrix = confusion_matrix(gt_categories, pred_categories)
    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(cf_matrix, annot=True, cbar=False, cmap='Blues', fmt='g', ax=ax)

    # Define your labels based on the confusion matrix size
    fold_labels = ['-45:-35', '-35:-25', '-25:-15', '-15:-5', '-5:5',
                   '5:15', '15:25', '25:35', '35:45']

    # Set the ticks positions and labels for both axes
    ax.set_xticks(np.arange(len(fold_labels)) + 0.5)  # Offset by 0.5 to center labels
    ax.set_yticks(np.arange(len(fold_labels)) + 0.5)

    ax.set_xticklabels(fold_labels, rotation=45, ha="right")
    ax.set_yticklabels(fold_labels, rotation=0)

    ax.set_title('HPE', loc='left', fontsize=16)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.savefig('yaw_heatmap.png')
    plt.show()
    # plt.xlabel('Predicted categories')
    # plt.ylabel('True categories')
    # plt.title('Confusion Matrix for Pitch Categorization')
    # plt.savefig('heatmap2.png')
    # plt.show()


gt_categories, pred_categories = evaluate(data_dir)
gt_categories = ['Unknown' if x is None else x for x in gt_categories]
pred_categories = ['Unknown' if x is None else x for x in pred_categories]

cf_matrix = confusion_matrix(gt_categories, pred_categories)

plot_confusion_matrix(gt_categories, pred_categories)
