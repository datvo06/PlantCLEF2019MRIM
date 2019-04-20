from __future__ import print_function, unicode_literals, division
import os
import pickle
import torch
import numpy as np
import math
from numpy.random import choice
from PIL import Image


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class PlantCLEFDataSet(object):
    def __init__(self, class_id_map, class_id_list_files, transform=None):
        '''Let's load all the files
        Args:
            class_id_map: a mapping from list of id to list of real class id
            class_id_list_files: the list of list of images files each inner
            list corresponding to a class's files
        '''
        self.class_id_map = class_id_map
        self.class_id_list_files = class_id_list_files
        self.class_lens = list(
            [len(class_id_list)
             for class_id_list in self.class_id_list_files]
        )
        random_drawing = np.array(self.class_lens)
        non_zero_class = random_drawing > 0
        class_probs = random_drawing[non_zero_class]/np.sum(
            random_drawing[non_zero_class])
        class_weights = np.ones_like(class_probs)/class_probs
        self.class_weights_normed = np.zeros_like(random_drawing)
        self.class_weights_normed[non_zero_class] = class_weights/np.sum(
            class_weights)
        self.transform = transform

    def __len__(self):
        return sum(self.class_lens)

    def __getitem__(self, idx):
        # Get a random image from a random class weighted
        class_id = choice(len(self.class_id_map), p=self.class_weights_normed)
        sampler_id = choice(len(self.class_id_list_files[class_id]))
        # Simply ignore sample if fail to open
        try:
            sample = np.array(
                Image.open(self.class_id_list_files[class_id][sampler_id]))
        except Exception:
            return None
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, class_id


def split_dataset(plant_clef_dataset, split_ratio_train=0.8):
    class_id_list_files = [class_list_file[:]
                           for class_list_file in
                           plant_clef_dataset.class_id_list_files]
    class_id_map = plant_clef_dataset.class_id_map[:]
    val_class_id_map = class_id_map[:]
    val_class_list_files = [[] for class_id in class_id_map]
    for each_class_id in class_id_list_files:
        # Get the class len
        num_samples = len(class_id_list_files[each_class_id])
        num_sample_val = math.ceil(num_samples*(1-split_ratio_train))
        # Pick out these samples
        chosen_vals = np.random.choice(num_samples, num_sample_val)
        # Sort the chosen val from largest to smallest
        chosen_vals = np.sort(chosen_vals)[::-1]
        for chosen_val in chosen_vals:
            val_class_list_files[each_class_id].append(
                class_id_list_files[each_class_id].pop(chosen_val))

    val_dataset = PlantCLEFDataSet(val_class_id_map,
                                   val_class_list_files,
                                   transform=plant_clef_dataset.transform)
    train_dataset = PlantCLEFDataSet(class_id_map,
                                     class_id_list_files,
                                     transform=plant_clef_dataset.transform)
    return train_dataset, val_dataset
