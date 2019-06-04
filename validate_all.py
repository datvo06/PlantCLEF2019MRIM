from __future__ import print_function
from __future__ import division
import sys
import glob
import os
import torch
import torch.nn as nn
import scipy as sp
import numpy as np
import torchvision
import pickle
from torchvision import models, transforms
from data_accessor import PlantCLEFDataSet
from PIL import Image
from quicktest_torchvision import initialize_model, my_collate
import torch.utils.data
from matplotlib import pyplot as plt


import time
import copy
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Top level data directory. Here we assume the format of the directory conforms
# to the ImageFolder structure

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "densenet"

# Number of classes in the dataset
num_classes = 10000

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False


def eval_model(model, dataloaders, criterion, dump_matrix=False):
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
    if dump_matrix:
        print("Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.4f}'.format(best_acc))
        pickle.dump(confusion_matrix, open('confusion_matrix_' +
                                           model_name + '.pkl', 'wb'))
    return model, val_acc_history, confusion_matrix


def ensemble_prediction_pooling(list_models, inputs, pool_op=np.mean):
    for i, model in enumerate(list_models):
        list_models[i] = model.to(device)
    inputs = inputs.to(device)
    print(inputs.size())
    outputs = []
    with torch.set_grad_enabled(False):
        for model in list_models:
            outputs.append(np.reshape(model(inputs).cpu().numpy(), (1, 10000, 1)))
    # Let's make average pool
    # outputs : B x C x 1
    # outputs = np.array(outputs)
    # print(outputs.shape)
    output = np.concatenate(outputs, axis=2)
    output = pool_op(output, axis=2)
    print(output.shape)
    return output


def ensemble_pooling(outputs, pool_op=np.mean):
    output = np.concatenate(outputs, axis=2)
    output = pool_op(output, axis=2)
    return output

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def observation_output(list_models, transforms, observation_file_paths, k):
    '''
    outputs: list of class probabilities
    '''
    output_each_samples = []
    for each_sample_path in observation_file_paths:
        inputs = Image.open(each_sample_path)
        inputs = transforms(inputs)
        inputs = torch.unsqueeze(inputs, 0)
        output_each_samples.append(
            np.reshape(ensemble_prediction_pooling(list_models, inputs, np.mean), (1, 10000, 1)))
    output = ensemble_pooling(output_each_samples, np.max)
    # Softmax
    output = output.flatten()
    output = softmax(output)
    classes_idx = output.argsort()[-k:][::-1]
    probs = output[classes_idx]
    print(classes_idx, probs)
    return classes_idx, probs


def predict_all_observation(
        list_models, transforms, observation_ids, prefix, class_id_map, k):
    observation_ids_results = {}
    for each_observation_id in observation_ids:
        observation_ids[each_observation_id] = [
            os.path.join(prefix, filepath)
            for filepath in observation_ids[each_observation_id]]
        class_idx, probs = observation_output(
            list_models, transforms, observation_ids[each_observation_id], k
        )
        class_original_ids = [class_id_map[class_id] for class_id in class_idx]
        observation_ids_results[each_observation_id] = (
            class_original_ids, probs
        )
    return observation_ids_results


def run_test(list_models, transforms, observation_ids, prefix, class_id_map, k,
             output_path='run.txt'):
    observation_ids_results = predict_all_observation(
        list_models, transforms, observation_ids, prefix, class_id_map, k)
    with open(output_path, 'w') as output_file:
        for each_observation_id in observation_ids_results:
            for i in range(k):
                output_file.write(
                    each_observation_id + ";" +
                    observation_ids_results[each_observation_id][0][i] + ";" +
                    str(observation_ids_results[each_observation_id][1][i]) +
                    ";" + str(i+1) + "\n")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Must point to a trainset and a model folder")
    trainset = pickle.load(open(sys.argv[1], 'rb'))
    model_lists = []
    model_dict_paths = glob.glob(os.path.join(sys.argv[2], '*.pth'))
    for model_dict_path in model_dict_paths:
        model_ft, input_size = initialize_model(
            model_name, num_classes,
            feature_extract, use_pretrained=False)
        model_dict = model_ft.state_dict()
        pretrained_dict = torch.load(model_dict_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict}
        model_ft.load_state_dict(pretrained_dict)
        model_ft = model_ft.to(device)
        model_lists.append(model_ft)
        model_ft.eval()

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    print("Initializing Datasets and Dataloaders...")
    observation_ids = pickle.load(open(sys.argv[3], 'rb'))
    run_test(model_lists, data_transform,
             observation_ids, sys.argv[4], trainset[0], 10)
    # evaluate
    # model_ft, hist, confusion_matrix = eval_model(model_ft, dataloaders_dict,
    #                                              criterion)
    # Sort the confusion matrix from lowest to highest
    '''
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
    '''
