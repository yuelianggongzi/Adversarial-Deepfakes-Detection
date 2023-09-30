"""
Create adversarial videos that can fool xceptionnet.

Usage:
python attack.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

built upon the code by Andreas Rössler for detecting deep fakes.
"""

import sys, os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image
from PIL import Image as pil_image
from tqdm import tqdm

from torch import autograd
import numpy
from torchvision import transforms

import json
import random
import imquality.brisque as brisque
import numpy as np

file_type_list=['jpg']

# I don't recommend this, but I like clean terminal output.
import warnings

warnings.filterwarnings("ignore")


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def test_video(video_path,video_name):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)
    fileName1=video_name+"dfdc.jpg"
    fileName2 = '/mnt/publicStoreA/videodata/dfdc-adv/all-keyframes-amp/'+video_name + "amp-face.jpg"
    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break

        height, width = image.shape[:2]
        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]
            img=cropped_face
            #cv2.imwrite(fileName2, img)
            # print('...step 1...')
            img= Image.fromarray(img[:, :, ::-1])
            #cv2.imwrite(fileName1, img)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            #f2shift = np.fft.ifftshift(np.angle(fshift))
            f2shift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f2shift)
            img_back = np.abs(img_back)
            img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
            img_back = img_back * 255
            img_back = img_back.astype(np.uint8)
            cv2.imwrite(fileName2, img_back)
            # print('...step 2...')
            break
        print('OK!')




if __name__ == '__main__':

    #video_path = '/mnt/publicStoreA/videodata/FFpp-original/deepfakes_raw'
    #video_path = '/mnt/publicStoreA/videodata/FFpp-adv/0/deepfakes'
    #video_path = '/mnt/publicStoreA/videodata/dfdc/dfdc_train_part_01'
    video_path = '/mnt/publicStoreA/videodata/dfdc-adv/all-keyframes'
    videos = os.listdir(video_path)

    f = open('1204.txt', 'w', encoding='utf-8')
    for video in videos:
        file_type=video.split('.'[-1])
        if(file_type[1] in file_type_list):

            # print(video)
            #video = 'fckxaqjbxk.avi'
            video_path_name = join(video_path, video)
            # blockPrint()
            # print(video_path_name)
            # if()
            result = test_video(video_path_name,video)
            result = str(video_path_name) + str(':') + str(result) + '\n'
            f.write(result)
            f.flush()
            # print('#######')
            print(result)

        # pbar_global.update(1)
    f.close()
