from PIL import Image
import os
import sys
import numpy as np
import shutil


train_dir = "/video/clef/LifeCLEF/PlantCLEF2019/train/"
if not os.path.exists(os.path.join(train_dir, "list_color_count_all")):
    os.makedirs(os.path.join(train_dir, "list_color_count_all"))
data_inspect_dir = os.path.join(train_dir, "list_color_count_all")


def check_is_image_plantCLEF(filepath):
    if filepath.endswith('xml'):
        return False
    return True


def get_list_files_filtered(root_directory, filter_func):
    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filter_func(filepath):
                yield filepath

def sorted_idx(the_list):
    indices = range(len(the_list))
    indices = list(sorted(indices, key=lambda index: the_list[index]))
    return indices


def detect_files_with_limited_num_colors(list_files):
    list_files_out = []
    list_count_out = []
    for each_file in list_files:
        try:
            img = np.array(Image.open(each_file))
            img = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
            num_unique_colors = np.unique(img, axis=0)
        except:
            continue
        if num_unique_colors.shape[0] < 6:
            print("Num unique_colors: ", num_unique_colors.shape[0])
            list_files_out.append(each_file)
            list_count_out.append(num_unique_colors.shape[0])
    return list_file_out, list_count_out


def copy_files(list_files, out_dir):
    list_file_out, list_count_out = detect_files_with_limited_num_colors(list_files)
    sorted_indices = sorted_idx(list_count_out)[::-1]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, 'list_files.txt'), 'w') as list_file_fp:
        for idx in sorted_indices:
            filepath = list_file_out[idx]
            count_out =  list_count_out[idx]
            list_file_fp.write(filepath + " - " + str(list_count_out) + "\n")
            filename = os.path.split(filepath)[1]
            shutil.copy(filepath, os.path.join(out_dir, filename))



if __name__ == '__main__':
    list_files = get_list_files_filtered(
            os.path.join(train_dir, 'data'),
            check_is_image_plantCLEF)
    copy_files(list_files, data_inspect_dir)
