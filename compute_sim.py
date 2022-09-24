# modified from https://github.com/Derekabc/CottonWeeds/tree/master/Image_Similarity
from __future__ import print_function, division
import os
from os import walk
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch.nn as nn
import torch
import random
from torchvision import transforms
from shutil import copyfile
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Test CottonWeed Classifier')
    parser.add_argument('--task', type=str, required=False,
                        default='deepweeds', help="choose from cottonweedid15 or deepweeds")
    parser.add_argument('--img_name', type=str, required=False,
                        default='/home/dong9/PycharmProjects/guided-diffusion/model256_deepweeds/samples_class_separate_ADMG/Negative/1830.jpg',
                        help="dir of image we want to study")
    parser.add_argument('--EVAL_DIR', type=str, required=False,
                        default='/home/dong9/PycharmProjects/guided-diffusion/datasets/DeepWeeds_train_gan/Snakeweed',
                        help="dir for the testing image")
    parser.add_argument('--seeds', type=int, required=False, default=0, help="random seed")
    parser.add_argument('--num_neighbor', type=int, required=False, default=3,
                        help="number of closest neighbors we want to find")
    parser.add_argument('--img_size', type=int, required=False, default=256, help="Image Size")
    args = parser.parse_args()
    return args


def cosine_distance(input1, input2):
    '''Calculating the distance of two inputs.

    The return values lies in [-1, 1]. `-1` denotes two features are the most unlike,
    `1` denotes they are the most similar.

    Args:
        input1, input2: two input numpy arrays.

    Returns:
        Element-wise cosine distances of two inputs.
    '''
    return np.dot(input1, input2.T) / \
           np.dot(np.linalg.norm(input1, axis=1, keepdims=True), \
                  np.linalg.norm(input2.T, axis=0, keepdims=True))


args = parse_args()
# for reproducing
torch.manual_seed(args.seeds)
torch.cuda.manual_seed(args.seeds)
torch.cuda.manual_seed_all(args.seeds) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args.seeds)
random.seed(args.seeds)
np.random.seed(args.seeds)


IMDIR = args.EVAL_DIR
if args.task == 'cottonweedid15':
    EVAL_MODEL = "inception_0.pth"  # for cottonweedid15
elif args.task == 'deepweeds':
    EVAL_MODEL = "inception_0_A.pth"  # for deepweeds
img_size = args.img_size

# Load the model for evaluation
model = torch.load(EVAL_MODEL).to("cuda")
model.dropout = nn.Identity()
model.fc = nn.Identity()
model.eval()


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

if args.task == 'cottonweedid15':
    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

    preprocess_save = transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.ToTensor()])
elif args.task == 'deepweeds':
    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    preprocess_save = transforms.Compose([
        transforms.ToTensor()])

# Enable gpu mode, if cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

neighbors = []
neighbor_sim = []

image_fake = Image.open(args.img_name).convert('RGB')
image_fake = preprocess(image_fake).to(device)

for (dirpath, dirnames, filenames) in walk(args.EVAL_DIR):
    for filename in filenames:
        image_real = Image.open(os.path.join(args.EVAL_DIR, filename))

        with torch.no_grad():
            # inception-v3
            x = model(image_fake[None, :])
            y = model(preprocess(image_real).to(device)[None, :])
            sim = cosine_distance(x.cpu(), y.cpu())[0][0]

        if len(neighbor_sim) < args.num_neighbor:
            neighbors.append(filename)
            neighbor_sim.append(sim)
            # from largest to smallest
            neighbors, neighbor_sim = zip(*sorted(zip(neighbors, neighbor_sim), reverse=True))
            neighbors, neighbor_sim = list(neighbors), list(neighbor_sim)
        else:
            for index, item in enumerate(neighbor_sim):
                if sim > item:
                    neighbor_sim[index] = sim
                    neighbors[index] = filename
                    break

if args.task == 'cottonweedid15':
    # save the neighbors to a folder
    os.makedirs('neighbors_cottonweedid15' + '/' + args.img_name.split("/")[-1].split('.')[0], exist_ok=True)
    copyfile(args.img_name,
             'neighbors_cottonweedid15' + '/' + args.img_name.split("/")[-1].split('.')[0] + '/' + args.img_name.split("/")[-1])

    for file in neighbors:
        image = Image.open(os.path.join(args.EVAL_DIR, file))
        # save_image(preprocess_save(image), 'neighbors' + '/' + args.img_name.split("/")[-1].split('.')[0] + '/' + file)
        # copyfile(center_crop_arr(image, 256), 'neighbors' + '/' + args.img_name.split("/")[-1].split('.')[0] + '/' + file)
        plt.imsave('neighbors_cottonweedid15' + '/' + args.img_name.split("/")[-1].split('.')[0] + '/' + file,
                   center_crop_arr(image, 256))

elif args.task == 'deepweeds':
    # save the neighbors to a folder
    os.makedirs('neighbors_deepweeds' + '/' + args.img_name.split("/")[-1].split('.')[0], exist_ok=True)
    copyfile(args.img_name,
             'neighbors_deepweeds' + '/' + args.img_name.split("/")[-1].split('.')[0] + '/' + args.img_name.split("/")[-1])

    for file in neighbors:
        image = Image.open(os.path.join(args.EVAL_DIR, file))
        # save_image(preprocess_save(image), 'neighbors' + '/' + args.img_name.split("/")[-1].split('.')[0] + '/' + file)
        # copyfile(center_crop_arr(image, 256), 'neighbors' + '/' + args.img_name.split("/")[-1].split('.')[0] + '/' + file)
        plt.imsave('neighbors_deepweeds' + '/' + args.img_name.split("/")[-1].split('.')[0] + '/' + file,
                   center_crop_arr(image, 256))
