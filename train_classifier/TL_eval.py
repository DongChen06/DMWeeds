import argparse
import os, csv


def parse_args():
    parser = argparse.ArgumentParser(description='Test CottonWeed Classifier')
    parser.add_argument('--model_name', type=str, required=False, default='densenet121',
                        help="choose a deep learning model")
    parser.add_argument('--EVAL_DIR', type=str, required=False,
                        default='/home/eb1228/PycharmProjects/pytorch_fid/dataset/DeepWeeds_test_gan',
                        help="dir for the testing image")
    parser.add_argument('--seeds', type=int, required=False, default=0, help="random seed")
    parser.add_argument('--device', type=int, required=False, default=0, help="GPU device")
    parser.add_argument('--batch_size', type=int, required=False, default=64, help="Training batch size")
    parser.add_argument('--img_size', type=int, required=False, default=256, help="Image Size")
    parser.add_argument('--use_weighting', type=bool, required=False, default=False, help="use weighted cross entropy or not")
    args = parser.parse_args()
    return args


args = parse_args()

import numpy as np
import torch
import random
from torchvision import datasets, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# for reproducing
torch.manual_seed(args.seeds)
torch.cuda.manual_seed(args.seeds)
torch.cuda.manual_seed_all(args.seeds) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args.seeds)
random.seed(args.seeds)
np.random.seed(args.seeds)


EVAL_DIR = args.EVAL_DIR
model_name = args.model_name
if args.use_weighting:
    EVAL_MODEL = './models/' + model_name + '_' + str(args.seeds) + '_w' + ".pth"
else:
    EVAL_MODEL = './models/' + model_name + '_' + str(args.seeds) + '_A' + ".pth"

# EVAL_MODEL = 'models_deepweeds_ori/vgg16_4_A.pth'
img_size = args.img_size
bs = args.batch_size

if not os.path.isfile('eval_performance.csv'):
    with open('eval_performance.csv', mode='w') as csv_file:
        fieldnames = ['Model', 'Evaluating Acc', 'Precision', 'Recall']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()

# Configure batch size and number of cpu's
num_cpu = multiprocessing.cpu_count()
# Prepare the eval data loader
eval_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

eval_dataset = datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
eval_loader = data.DataLoader(eval_dataset, batch_size=bs, shuffle=True,
                              num_workers=num_cpu, pin_memory=True)

# Enable gpu mode, if cuda available
if args.device == 0:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
elif args.device == 1:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Number of classes and dataset-size
num_classes = len(eval_dataset.classes)
dsize = len(eval_dataset)

# deepweeds
class_names = ['Chineeapple', 'Lantana', 'Parkinsonia', 'Parthenium',
               'Pricklyacacia', 'Rubbervine', 'Siamweed', 'Snakeweed', 'Negative']

# cottonweeds
# class_names = ['Carpetweeds', 'Eclipta', 'Goosegrass', 'Morningglory',
#                'Nutsedge', 'PalmerAmaranth', 'Purslane', 'Sicklepod', 'SpottedSpurge', 'Waterhemp']

# Initialize the prediction and label lists
predlist = torch.zeros(0, dtype=torch.long, device='cpu')
lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

# Evaluate the model accuracy on the dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predlist = torch.cat([predlist, predicted.view(-1).cpu()])
        lbllist = torch.cat([lbllist, labels.view(-1).cpu()])

# Overall accuracy
overall_accuracy = 100 * correct / total
print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize, overall_accuracy))

# Confusion matrix
conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
print('Confusion Matrix')
print('-' * 16)
print(conf_mat, '\n')


precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
print('-' * 16)
print("Precision", np.mean(precision))
print("Recall", np.mean(recall), '\n')

if not os.path.exists('Confusing_Matrices'):
    os.mkdir('Confusing_Matrices')
if not os.path.exists('Confusing_Matrices/plots/'):
    os.mkdir('Confusing_Matrices/plots/')
if not os.path.exists('Confusing_Matrices/csv/'):
    os.mkdir('Confusing_Matrices/csv/')


plt.figure(figsize=(10, 6))
df_cm = pd.DataFrame(conf_mat, index=class_names,
                     columns=class_names)
sn.set(font_scale=1.0)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap='Greens')
plt.xticks(rotation=75, fontsize=14)
plt.tight_layout()
if args.use_weighting:
    plt.savefig('Confusing_Matrices/plots/' + model_name + '_cm_' + str(args.seeds) + '_w.png')
else:
    plt.savefig('Confusing_Matrices/plots/' + model_name + '_cm_' + str(args.seeds) + '.png')
# plt.show()

if args.use_weighting:
    df_cm.to_csv('Confusing_Matrices/csv/' + model_name + '_cm_' + str(args.seeds) + '_w.csv')
else:
    df_cm.to_csv('Confusing_Matrices/csv/' + model_name + '_cm_' + str(args.seeds) + '.csv')

# Per-class accuracy
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
print('Per class accuracy')
print('-' * 18)
for label, accuracy in zip(eval_dataset.classes, class_accuracy):
    print('Accuracy of class %8s : %0.2f %%' % (label, accuracy))

with open('eval_performance.csv', 'a+', newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    csv_writer.writerow([model_name, overall_accuracy, np.mean(precision), np.mean(recall)])
