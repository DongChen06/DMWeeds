# modified from https://github.com/sbarratt/inception-score-pytorch
from __future__ import print_function, division
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

import numpy as np
from scipy.stats import entropy


def inception_score(dataloader, cuda=True, batch_size=32, resize=False, splits=1, num_classes=15):
    """Computes the inception score of the generated images

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    num_classes: number of weed classes
    """
    N = len(dataloader.dataset)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dev = torch.device("cuda")
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor
        dev = torch.device("cpu")

    inception_model = torch.load(PATH).to(dev)
    inception_model.eval()

    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, num_classes))

    for i, batch in enumerate(dataloader, 0):
        batch = batch[0].type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    from torchvision import datasets, transforms
    import torch.utils.data as data

    # PATH = 'inception_0.pth' # for cottonweedid15
    # num_classes = 15

    PATH = 'inception_0_A.pth' # for deepweeds
    num_classes = 9
    EVAL_DIR = '/home/dong9/PycharmProjects/PyTorch-StudioGAN/images/samples/DeepWeeds-StyleGAN3-r-paper-train-2022_08_30_14_24_00/fake'

    # Prepare the eval data loader
    img_size = 256
    bs = 32
    num_cpu = 8

    eval_transform = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.CenterCrop(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    eval_dataset = datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
    dataloader = data.DataLoader(eval_dataset, batch_size=bs, shuffle=True,
                                  num_workers=num_cpu, pin_memory=True)

    print ("Calculating Inception Score...")
    print(inception_score(dataloader, cuda=True, batch_size=bs, resize=True, splits=10, num_classes=num_classes))
