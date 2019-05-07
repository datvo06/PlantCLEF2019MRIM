from __future__ import print_function, unicode_literals, division
import os
import pickle
import torch
import numpy as np
import math
from numpy.random import choice
from PIL import Image
import random


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class PlantCLEFDataSetWeightOversamp(object):
    def __init__(self, class_id_map, class_id_list_files,
                 transform=None,
                 is_train=False, prefix='', gamma=0.5):
        '''Let's load all the files
        Args:
            class_id_map: a mapping from list of id to list of real class id
            class_id_list_files: the list of list of images files each inner
            list corresponding to a class's files
        '''
        self.class_id_map = class_id_map
        self.class_id_list_files = class_id_list_files
        self.prefix = prefix
        inverse_class_id_map = list(
            [(class_id[1], class_id[0]) for class_id in class_id_map.items()])
        self.inverse_class_id_map = dict(inverse_class_id_map)

        self.class_lens = list(
            [len(class_id_list)
             for class_id_list in self.class_id_list_files]
        )
        # Let's sort classes from lowest to highest
        # Then resample it so that each class at least a minimum number of
        # samples
        self.transform = transform
        self.class_lens = np.array(self.class_lens)
        self.oversampling_factors = []
        self.weight_factors = []
        if is_train:
            self.m = np.median(self.class_lens)
            for class_id in range(len(self.class_id_list_files)):
                if len(self.class_id_list_files[class_id]) == 0:
                    continue
                if len(self.class_id_list_files[class_id]) < self.m:
                    self.oversampling_factors.append(
                        (1 + self.m/self.class_lens[class_id])/2)
                else:
                    self.oversampling_factors.append(1)
                self.weight_factors.append(
                    math.pow(
                        self.oversampling_factors[class_id] *
                        self.class_lens[class_id],
                        gamma - 1))
        # Renormalize weight factors
        self.weight_factors = np.array(self.weight_factors)
        self.oversampling_factors = np.array(self.oversampling_factors)
        sum_total = np.sum(
            self.weight_factors*self.oversampling_factors*self.class_lens)
        normalizing_factor = sum_total / np.sum(self.class_lens)
        self.weight_factors /= normalizing_factor
        self.class_lens = np.array(self.weight_factors *\
            self.oversampling_factors * self.class_lens).astype(np.int)
        # First time shuffling
        # Flatten them all then shuffle
        self.filepath_merged = []
        for each_class_id in range(len(self.class_id_list_files)):
            self.filepath_merged.extend(
                self.class_id_list_files[each_class_id])
        random.shuffle(self.filepath_merged)
        # Shuffle them up
        self.reshuffle()

    def reshuffle(self):
        self.temp_class_list_files = [
            [] for i in range(len(self.class_id_list_files))]
        for class_id in self.class_id_list_files:
            repeat = math.ceil(
                self.class_lens[class_id]/len(
                    self.class_id_list_files[class_id])
            )
            if len(self.class_id_list_files[class_id]) == 0:
                continue
            self.temp_class_list_files[class_id] =\
                self.class_id_list_files[class_id] * repeat
            self.temp_class_list_files[class_id] =\
                self.temp_class_list_files[class_id][
                    :self.class_lens[class_id]]
        self.filepath_merged = []
        for each_class_id in range(len(self.temp_class_list_files)):
            self.filepath_merged.extend(
                self.temp_class_list_files[each_class_id])
        random.shuffle(self.filepath_merged)
        if len(self.prefix) > 0:
            self.filepath_merged = [os.path.join(self.prefix, filepath[41:])
                                    for filepath in self.filepath_merged]

    def __len__(self):
        return np.sum(self.class_lens)

    def get_random_sample(self):
        # Get a random image from a random class weighted
        class_id = choice(self.non_zero_index.shape[0],
                          p=self.class_weights_normed)
        sampler_id = choice(
            len(self.class_id_list_files[self.non_zero_index[class_id]]))
        # Simply ignore sample if fail to open
        try:
            sample = Image.open(
                self.class_id_list_files[
                    self.non_zero_index[class_id]
                ][sampler_id]).convert('RGB')
        except Exception:
            return None
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.non_zero_index[class_id]

    def __getitem__(self, idx):
        filepath = self.filepath_merged[idx]
        try:
            sample = Image.open(filepath).convert('RGB')
        except Exception:
            return None
        if self.transform is not None:
            sample = self.transform(sample)

        class_name = os.path.basename(os.path.dirname(filepath))
        class_id = self.inverse_class_id_map[class_name]
        return sample, class_id


class PlantCLEFDataSet(object):
    def __init__(self, class_id_map, class_id_list_files,
                 transform=None,
                 is_train=False, min_samples=40):
        '''Let's load all the files
        Args:
            class_id_map: a mapping from list of id to list of real class id
            class_id_list_files: the list of list of images files each inner
            list corresponding to a class's files
        '''
        self.class_id_map = class_id_map
        self.class_id_list_files = class_id_list_files
        inverse_class_id_map = list(
            [(class_id[1], class_id[0]) for class_id in class_id_map.items()])
        self.inverse_class_id_map = dict(inverse_class_id_map)

        self.class_lens = list(
            [len(class_id_list)
             for class_id_list in self.class_id_list_files]
        )
        # Let's sort classes from lowest to highest
        # Then resample it so that each class at least a minimum number of
        # samples
        self.transform = transform
        self.class_lens = np.array(self.class_lens)
        if is_train:
            for class_id in range(len(self.class_id_list_files)):
                if len(self.class_id_list_files[class_id]) == 0:
                    continue
                if len(self.class_id_list_files[class_id]) < min_samples:
                    repeat = math.ceil(
                        min_samples/self.class_lens[class_id])
                    self.class_id_list_files[class_id] =\
                        self.class_id_list_files[class_id] * repeat
                    self.class_id_list_files[class_id] =\
                        self.class_id_list_files[class_id][:min_samples]
        self.class_lens = list(
            [len(class_id_list)
             for class_id_list in self.class_id_list_files]
        )

        # Flatten them all then shuffle
        self.filepath_merged = []
        for each_class_id in range(len(self.class_id_list_files)):
            self.filepath_merged.extend(
                self.class_id_list_files[each_class_id])
        random.shuffle(self.filepath_merged)
        # Shuffle them up
        '''
        random_drawing = np.array(self.class_lens)
        self.non_zero_index = np.where(self.class_lens > 0)[0]
        non_zero_class = self.class_lens > 0
        class_probs = random_drawing[non_zero_class]/np.sum(
            random_drawing[non_zero_class])
        class_weights = np.ones_like(class_probs)/class_probs
        self.class_weights_normed = class_weights/np.sum(
            class_weights)
        print("Number of empty classes: ",
              len(self.class_id_list_files) - self.non_zero_index.shape[0])
              '''

    def __len__(self):
        return len(self.filepath_merged)

    def get_random_sample(self):
        # Get a random image from a random class weighted
        class_id = choice(self.non_zero_index.shape[0],
                          p=self.class_weights_normed)
        sampler_id = choice(
            len(self.class_id_list_files[self.non_zero_index[class_id]]))
        # Simply ignore sample if fail to open
        try:
            sample = Image.open(
                self.class_id_list_files[
                    self.non_zero_index[class_id]
                ][sampler_id]).convert('RGB')
        except Exception:
            return None
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.non_zero_index[class_id]

    def __getitem__(self, idx):
        filepath = self.filepath_merged[idx]
        try:
            sample = Image.open(filepath).convert('RGB')
        except Exception:
            return None
        if self.transform is not None:
            sample = self.transform(sample)

        class_name = os.path.basename(os.path.dirname(filepath))
        class_id = self.inverse_class_id_map[class_name]
        return sample, class_id



def split_dataset(plant_clef_dataset, split_ratio_train=0.8):
    class_id_list_files = [class_list_file[:]
                           for class_list_file in
                           plant_clef_dataset.class_id_list_files]
    class_id_map = dict(plant_clef_dataset.class_id_map.items())
    val_class_id_map = dict(class_id_map.items())
    val_class_list_files = [[] for class_id in class_id_map]
    num_empty = 0
    for each_class_id in range(len(class_id_list_files)):
        # Get the class len
        num_samples = len(class_id_list_files[each_class_id])
        if num_samples == 0:
            num_empty += 1
            continue
        num_sample_val = math.ceil(num_samples*(1-split_ratio_train))
        # Pick out these samples
        shuffled = np.arange(num_samples)
        np.random.shuffle(shuffled)
        chosen_vals = shuffled[:num_sample_val]
        # Sort the chosen val from largest to smallest
        chosen_vals = np.sort(chosen_vals)[::-1]
        for chosen_val in chosen_vals:
            val_class_list_files[each_class_id].append(
                class_id_list_files[each_class_id].pop(chosen_val))
    print("Number of empty class: ", num_empty)

    val_dataset = PlantCLEFDataSet(val_class_id_map,
                                   val_class_list_files,
                                   transform=plant_clef_dataset.transform)
    train_dataset = PlantCLEFDataSet(class_id_map,
                                     class_id_list_files,
                                     transform=plant_clef_dataset.transform,
                                     is_train=True)
    return train_dataset, val_dataset


def remove_file_by_names(class_id_map, class_id_list_files, list_files,
                         avoid_empty_class=False):
    list_files_names = list([filename.split(".")[0]
                             for filename in list_files])

    list_files_names = sorted(list_files_names)
    inverse_class_id_map = list([(class_id[1], class_id[0])
                                 for class_id in class_id_map.items()])

    inverse_class_id_map = dict(inverse_class_id_map)
    class_lens = [len(class_id_list_files[class_id])
                  for class_id in range(len(class_id_list_files))]
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
                file_par_dir = os.path.basename(os.path.dirname(
                    filepath_merged[filename_sorted_idx[curr_idx]])
                )
                original_class_id = inverse_class_id_map[file_par_dir]
                if class_lens[original_class_id] == 1:
                    curr_idx += 1
                    continue
                class_lens[original_class_id] -= 1
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


def remove_all_multiples_no_empty(class_id_map,
                                  class_id_list_files, list_multiples):
    ''''''
    # Sorted by number of multipliers
    list_removes = [filepair for filepair in list_multiples if
                    filepair[1] > 1]
    list_removes = sorted(list_removes, key=lambda filepair: filepair[1])[::-1]
    list_files = [filepair[0] for filepair in list_removes]
    class_id_list_files_new = remove_file_by_names(
        class_id_map, class_id_list_files, list_files, True)
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


def get_list_multiples(class_id_map, class_id_list_files):
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
    last_file_name = filename_merged[filename_sorted_idx[0]]
    list_multiples = [[last_file_name, 1]]
    for index in filename_sorted_idx[1:]:
        if filename_merged[index] == last_file_name:
            list_multiples[-1][1] += 1
        else:
            last_file_name = filename_merged[index]
            list_multiples.append([last_file_name, 1])
    return list_multiples


def get_common_categories(class_id_map1, class_id_map2):
    class_names_1 = [pair[1] for pair in class_id_map1.items()]
    class_names_2 = [pair[1] for pair in class_id_map2.items()]
    # Sort by names
    class_names_1 = sorted(class_names_1)
    class_names_2 = sorted(class_names_2)
    curr_index = 0
    commons = []
    for name in class_names_2:
        if curr_index == len(class_names_1) - 1:
            break
        while(curr_index < (len(class_names_1) -1) and
              class_names_1[curr_index] < name):
            curr_index += 1
        if class_names_1[curr_index] == name:
            commons.append(name)
            continue
    return commons


def remap_categories(trainset1, trainset2):
    pass


def remodel_distribution(dataset1, dataset2):
    '''Remodel dataset1 samples distribution to that of dataset2
    dataset: class_id_map and list files
    '''
    class_id_map1, class_files1 = dataset1
    class_id_map2, class_files2 = dataset2
    assert len(class_files1) == len(class_files2), \
        "Number of classes must be equal"
    assert all(len(class_file_list) != 0 for class_file_list in class_files1),\
        "Number of samples must be atleast 1"
    # First, get all class lens
    class_lens1 = [len(class_file_list) for class_file_list in class_files1]
    class_lens2 = [len(class_file_list) for class_file_list in class_files2]
    # Then sort them
    class_len_id_sorted1 = sorted(range(len(class_lens1)),
                                  key=lambda i: class_lens1[i])
    class_len_id_sorted2 = sorted(range(len(class_lens1)),
                                  key=lambda i: class_lens2[i])
    # Get the inverse map also
    # inv_maps1 = dict([(item[1], item[0]) for item in class_id_map1.items()])
    # inv_maps2 = dict([(item[1], item[0]) for item in class_id_map2.items()])
    new_class_files = [[] for i in range(len(class_files1))]
    for i, original_id1 in enumerate(class_len_id_sorted1):
        # Get this class len
        len1 = class_lens1[original_id1]
        len2 = class_lens2[class_len_id_sorted2[i]]
        if len1 < len2:
            # If len 1 < len 2: repeat
            repeat = math.ceil(float(len2)/len1)
            new_class_files[class_len_id_sorted1[i]] = (
                class_files1[class_len_id_sorted1[i]][:]*repeat)[:len2]
        else:
            # Else shuffle, sample
            new_class_files[class_len_id_sorted1[i]] =\
                class_files1[class_len_id_sorted1[i]][:]
            random.shuffle(new_class_files[class_len_id_sorted1[i]])
            new_class_files[class_len_id_sorted1[i]] =\
                new_class_files[class_len_id_sorted1[i]][:len2]
    return class_id_map1, new_class_files
