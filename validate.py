from __future__ import print_function
from __future__ import division
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import pickle
from torchvision import datasets, models, transforms
from shufflenet import ShuffleNet
from quicktest_torchvision import initialize_model, MyImageFolder, my_collate
import torch.utils.data
from matplotlib import pyplot as plt


import time
import copy
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Top level data directory. Here we assume the format of the directory conforms
# to the ImageFolder structure
data_dir = "/video/clef/LifeCLEF/PlantCLEF2019/train/data"
data_dir_web = "/video/clef/LifeCLEF/PlantCLEF2017/web/data"


# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "densenet"

# Number of classes in the dataset
num_classes = 8500

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False


def eval_model(model, dataloaders, criterion):
    global model_name
    since = time.time()
    val_acc_history = []
    best_acc = 0.0
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Each epoch has a training and validation phase
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    running_samples = 0
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        running_loss += loss.item()*inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        running_samples += np.prod(list(labels.data.size()))

    epoch_loss = running_loss / len(dataloaders['val'].dataset)
    epoch_acc = running_corrects.double() / len(
        dataloaders['val'].dataset
    )

    print('{} loss: {:.4f} Acc: {:.4f}'.format(
        'val', epoch_loss, epoch_acc)
    )
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    pickle.dump(confusion_matrix, open('confusion_matrix_' +
                                       model_name + '.pkl', 'wb'))
    return model, val_acc_history, confusion_matrix


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def plot_cm(cm, title='reduced confusion matrix', normalize=False):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    classes = np.arange(cm.shape[0])
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == '__main__':
    model_ft, input_size = initialize_model(
        model_name, num_classes,
        feature_extract, use_pretrained=False)
    if (len(sys.argv) >= 2):
        try:
            model_dict = model_ft.state_dict()
            pretrained_dict = torch.load(sys.argv[1])
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict}
            model_ft.load_state_dict(pretrained_dict)
        except:
            pretrained_dict = torch.load(sys.argv[1])
            model_ft = models.densenet201(pretrained=False)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, 10000)
            model_dict = model_ft.state_dict()
            pretrained_dict = torch.load(sys.argv[1])
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict}
            model_ft.load_state_dict(pretrained_dict)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    print(model_ft)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: MyImageFolder(data_dir, data_transforms[x])
                      for x in ['val']}
    dataloaders_dict = {
        'val': torch.utils.data.DataLoader(
                image_datasets['val'],
                batch_size=batch_size,
                num_workers=4, collate_fn=my_collate
                )
        }

    # Initialize the model for this run

    # Print the model we just instantiated

    # Send the model to GPU
    model_ft = model_ft.to(device)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist, confusion_matrix = eval_model(model_ft, dataloaders_dict,
                                                  criterion)
    # Sort the confusion matrix from lowest to highest
    num_samples_per_classes = np.sum(confusion_matrix, axis=0)
    indices_sum = np.argsort(num_samples_per_classes)
    confusion_matrix = confusion_matrix[indices_sum, :]
    reduced_num_samples = np.zeros(3)
    reduced_confusion_matrix = np.zeros((3, 3))
    unit_span = int(num_classes/3)
    for i in range(3):
        reduced_num_samples = np.sum(num_samples_per_classes[:i*unit_span])
        for j in range(3):
            reduced_confusion_matrix[i, j] = np.sum(
                confusion_matrix[i*unit_span:(i+1)*unit_span,
                j*unit_span:(j+1)*unit_span])
    reduced_num_samples = np.expand_dims(reduced_num_samples, 0)
    reduced_confusion_matrix /= reduced_num_samples
    plot_cm(reduced_confusion_matrix)
    plt.savefig('confusion_matrix_' + model_name + '.jpg')
