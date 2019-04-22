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
    for each_class_id in range(len(class_id_list_files)):
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


def remove_file_by_names(class_id_map, class_id_list_files, list_files):
    list_files_names = list([filename.split(".")[0]
                             for filename in list_files])

    list_files_names = sorted(list_files_names)
    inverse_class_id_map = list([(class_id[1], class_id[0])
                                 for class_id in class_id_map.items()])

    inverse_class_id_map = dict(inverse_class_id_map)
    # Sort it, N log N

    # Merge all class file names
    # O(N)
    filepath_merged = []
    for each_class_id in range(len(class_id_list_files)):
        filepath_merged.extend(class_id_list_files[each_class_id])
    # O(N)
    # Get a new list with names only
    filename_merged = [os.path.split(filepath)[1].split(".")[0]
                       for filepath in filepath_merged]
    # O(N log N)
    filename_sorted_idx = sorted(range(len(filename_merged)),
                                 key=lambda idx: filename_merged[idx])
    curr_idx = 0
    deleting_idx = []
    # O(N)
    for index, filename in enumerate(list_files_names):
        while(curr_idx < len(filename_merged) and
              filename_merged[filename_sorted_idx[curr_idx]] < filename
              ):
            curr_idx += 1
        if filename_merged[filename_sorted_idx[curr_idx]] == filename:
            duplicated = 0
            # print(index)
            while(
                curr_idx < len(filename_merged) and
                filename_merged[filename_sorted_idx[curr_idx]] == filename
                  ):
                deleting_idx.append(filename_sorted_idx[curr_idx])
                duplicated += 1
                curr_idx += 1
    print("list file names deleted: ", len(deleting_idx))
    # O(NlogN)
    deleting_idx = sorted(deleting_idx)[::-1]  # From top down
    # O(N)
    for idx in deleting_idx:
        filepath_merged.pop(idx)

    # Mapping back, O(N)
    class_id_list_files_new = [[] for i in range(len(inverse_class_id_map))]
    for filepath in filepath_merged:
        # file_dir, filename = os.path.split(filepath)
        file_par_dir = os.path.basename(os.path.dirname(filepath))
        class_id_list_files_new[inverse_class_id_map[file_par_dir]].append(
            filepath)
    return class_id_list_files_new


def remove_multiples(class_id_map, class_id_list_files, list_multiples,
                     maximum_allowed=1):
    '''
    list multiples: (filename, num_repeat)
    '''
    # Let's search for each of the class and remove all duplicated ids
    list_removes = [filepair[0] for filepair in list_multiples if
                    filepair[1] > maximum_allowed]
    print(len(list_removes))
    class_id_list_files_new = remove_file_by_names(
        class_id_map, class_id_list_files, list_removes)
    # Just for debugging
    print("Number of files removed: ",
          sum([len(class_id_files) for class_id_files in class_id_list_files])
          - sum([len(class_id_files) for class_id_files in
                 class_id_list_files_new]))
    return class_id_list_files_new


def get_list_removed_files(src_dir, dst_dir):
    list_files_src = []
    list_files_dst = []
    for root, dirname, filenames in os.walk(src_dir):
        for filename in filenames:
            list_files_src.append(os.path.join(root, filename))
    for root, dirname, filenames in os.walk(dst_dir):
        for filename in filenames:
            list_files_dst.append(os.path.join(root, filename))
    filename_merged_src = [os.path.split(filepath)[1].split(".")[0]
                           for filepath in list_files_src]
    filename_merged_dst = [os.path.split(filepath)[1].split(".")[0]
                           for filepath in list_files_dst]
    filename_merged_dst = sorted(filename_merged_dst)

    # O(N log N)
    filename_sorted_idx = sorted(range(len(filename_merged_src)),
                                 key=lambda idx: filename_merged_src[idx])
    curr_idx = 0
    list_removeds = []
    # O(N)
    for filename in filename_merged_dst:
        while(curr_idx < len(filename_merged_src) and
              filename_merged_src[filename_sorted_idx[curr_idx]] < filename):
            list_removeds.append(
                filename_merged_src[filename_sorted_idx[curr_idx]])
            curr_idx += 1
        if filename_merged_src[filename_sorted_idx[curr_idx]] == filename:
            while(
               curr_idx < len(filename_merged_src) and
               filename_merged_src[filename_sorted_idx[curr_idx]] == filename
            ):
                curr_idx += 1
    print("List removed len: ", len(list_removeds))
    return list_removeds
