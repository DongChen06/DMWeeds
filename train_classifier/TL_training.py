"""
Training weed classification models with transfer learning
Dong Chen (chendon9@msu.edu, 2022-07-23)
"""
import csv
import argparse
from efficientnet_pytorch import EfficientNet
import pretrainedmodels  # for inception-v4 and xception


def parse_args():
    parser = argparse.ArgumentParser(description='Train CottonWeed Classifier')
    parser.add_argument('--train_directory', type=str, required=False,
                        default='/home/eb1228/PycharmProjects/pytorch_fid/dataset/dataset/CottonWeeds_wo_cleanup',
                        help="training directory")
    parser.add_argument('--valid_directory', type=str, required=False,
                        default='/home/eb1228/PycharmProjects/pytorch_fid/dataset/dataset/CottonWeeds_wo_cleanup',
                        help="validation directory")
    parser.add_argument('--model_name', type=str, required=False,
                        default='resnet50', help="deep learning model")
    parser.add_argument('--train_mode', type=str, required=False, default='finetune',
                        help="Set training mode: finetune, transfer, scratch")
    parser.add_argument('--num_classes', type=int, required=False, default=10, help="Number of Classes")
    parser.add_argument('--seeds', type=int, required=False, default=0, help="random seed")
    parser.add_argument('--is_augmentation', type=bool, required=False, default=True,
                        help="use data augmentation or not")
    parser.add_argument('--device', type=int, required=False, default=0,
                        help="GPU device")
    parser.add_argument('--epochs', type=int, required=False, default=70, help="Training Epochs")
    parser.add_argument('--batch_size', type=int, required=False, default=32, help="Training batch size")
    parser.add_argument('--img_size', type=int, required=False, default=256, help="Image Size")
    parser.add_argument('--use_weighting', type=bool, required=False, default=False, help="use weighted cross entropy or not")
    args = parser.parse_args()
    return args


args = parse_args()
import torch, os
import random
import numpy as np

# for reproducing
torch.manual_seed(args.seeds)
torch.cuda.manual_seed(args.seeds)
torch.cuda.manual_seed_all(args.seeds) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args.seeds)
random.seed(args.seeds)
np.random.seed(args.seeds)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(args.seeds)


from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchsummary import summary
import time, copy

num_classes = args.num_classes
train_mode = args.train_mode
model_name = args.model_name
num_epochs = args.epochs
bs = args.batch_size
img_size = args.img_size
train_directory = args.train_directory + '/DATA_{}'.format(args.seeds) + '/train'
valid_directory = args.valid_directory + '/DATA_{}'.format(args.seeds) + '/val'

if not os.path.isfile('train_performance.csv'):
    with open('train_performance.csv', mode='w') as csv_file:
        fieldnames = ['Index', 'Model', 'Training Time', 'Trainable Parameters', 'Best Train Acc', 'Best Train Epoch',
                      'Best Val Acc', 'Best Val Epoch']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

# Set the model save path
if args.use_weighting:
    print(True)
    if args.is_augmentation:
        PATH = 'models/' + model_name + "_" + str(args.seeds) + "_wA" + ".pth"
    else:
        PATH = 'models/' + model_name + "_" + str(args.seeds) + "_w" + ".pth"
else:
    if args.is_augmentation:
        PATH = 'models/' + model_name + "_" + str(args.seeds) + "_A" + ".pth"
    else:
        PATH = 'models/' + model_name + "_" + str(args.seeds) + ".pth"

if not os.path.exists('models'):
    os.mkdir('models')

# Number of workers
num_cpu = 8  # multiprocessing.cpu_count()

# Applying transforms to the data
if args.is_augmentation:
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(360),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

else:
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=img_size),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }


# Load data from folders
dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

# Size of train and validation data
dataset_sizes = {
    'train': len(dataset['train']),
    'valid': len(dataset['valid'])
}

# Create iterators for data loading
dataloaders = {
    'train': data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                             num_workers=num_cpu, pin_memory=True, drop_last=True,
                             worker_init_fn=seed_worker, generator=g),
    'valid': data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                             num_workers=num_cpu, pin_memory=True, drop_last=True,
                             worker_init_fn=seed_worker, generator=g)}

# Class names or target labels
class_names = dataset['train'].classes
print("Classes:", class_names)

# Print the train and validation data sizes
print("Training-set size:", dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'])

print("\nLoading pretrained-model for finetuning ...\n")
if model_name == 'inception':
    # args.img_size = 299
    model_ft = models.inception_v3(pretrained=True)
    model_ft.aux_logits = False
    model_ft.AuxLogits = None
    # Handle the auxilary net
    # num_ftrs = model_ft.AuxLogits.fc.in_features
    # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
elif model_name == 'resnet50':
    # Modify fc layers to match num_classes
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
elif model_name == 'efficientnet-b3':
    model_ft = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
elif model_name == 'vgg16':
    model_ft = models.vgg16(pretrained=True)
    model_ft.classifier[6] = nn.Linear(4096, num_classes)
elif model_name == 'densenet121':
    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
elif model_name == 'dpn68':
    model_ft = pretrainedmodels.dpn68(pretrained='imagenet')
    model_ft.last_linear = nn.Conv2d(832, num_classes, kernel_size=(1, 1), stride=(1, 1))

# Transfer the model to GPU
if args.device == 0:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
elif args.device == 1:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
# model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)

# Print model summary
print('Model Summary:-\n')
for num, (name, param) in enumerate(model_ft.named_parameters()):
    print(num, name, param.requires_grad)
if model_name == 'inception':
    summary(model_ft, input_size=(3, 299, 299))
elif model_name == 'densenet121' or 'densenet161' or 'resnext50_32x4d' or 'resnext101_32x8d':
    pass
else:
    summary(model_ft, input_size=(3, img_size, img_size))
print(model_ft)

# for class unbalance
weights = np.ones(args.num_classes)
class_weight = torch.FloatTensor(list(weights)).to(device)

pytorch_total_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
# print("Total parameters:", pytorch_total_params)
# Loss function
criterion = nn.CrossEntropyLoss(weight=class_weight)

# Optimizer
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

# Learning rate decay
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=150, gamma=0.9)

# Model training routine
print("\nTraining:-\n")


def train_model(model, criterion, optimizer, scheduler, num_epochs=150):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc = 0.0
    best_train_epoch = 0
    best_val_epoch = 0
    best_val_acc = 0.0

    if args.use_weighting:
        # Tensorboard summary
        if args.is_augmentation:
            writer = SummaryWriter(log_dir=('./runs/' + model_name + '_wA' + '/' + str(args.seeds)))
        else:
            writer = SummaryWriter(log_dir=('./runs/' + model_name + '_w' + '/' + str(args.seeds)))
    else:
        if args.is_augmentation:
            writer = SummaryWriter(log_dir=('./runs/' + model_name + '/' + str(args.seeds) + '_A'))
        else:
            writer = SummaryWriter(log_dir=('./runs/' + model_name + '/' + str(args.seeds)))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
                if epoch_acc > best_train_acc:
                    best_train_acc = epoch_acc
                    best_train_epoch = epoch
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_val_epoch = epoch
        print()

    time_elapsed = time.time() - since

    with open('train_performance.csv', 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow([args.seeds, model_name, '{:.0f}m'.format(
            time_elapsed // 60), pytorch_total_params, '{:4f}'.format(best_train_acc.cpu().numpy()),
                             best_train_epoch, '{:4f}'.format(best_val_acc.cpu().numpy()), best_val_epoch])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
# Save the entire model
print("\nSaving the model...")
torch.save(model_ft, PATH)