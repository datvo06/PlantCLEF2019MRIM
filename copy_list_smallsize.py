import os
import sys
import shutil

train_dir = "/video/clef/LifeCLEF/PlantCLEF2019/train/"
if not os.path.exists(os.path.join(train_dir, "store_temp_small_size_2")):
    os.makedirs(os.path.join(train_dir, "store_temp_small_size_2"))
data_inspect_dir = os.path.join(train_dir, "store_temp_small_class")

def check_small_size(filepath, size_in_bytes_min = 0, size_in_bytes_max=1024):
    if os.path.getsize(filepath) <= size_in_bytes_max and os.path.getsize(filepath) > size_in_bytes_min:
        return True
    return False


def check_is_image_plantCLEF(filepath):
    if filepath.endswith('xml'):
        return False
    return True


def get_list_files_filtered(root_directory, filter_func):
    list_files = []
    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filter_func(filepath):
                list_files.append(filepath)
    return list_files


def copy_small_images(train_dir, dest, size_in_bytes_min, size_in_bytes_max):
    list_files = get_list_files_filtered(
            os.path.join(train_dir, 'data'),
            lambda filepath: check_is_image_plantCLEF(filepath) and\
                    check_small_size(filepath, size_in_bytes_min, size_in_bytes_max))
    print("Total number of files: ", len(list_files))
    if not os.path.exists(dest):
        os.makedirs(dest)
    with open(os.path.join(dest, 'list_small_size_' + str(size_in_bytes_min) + '_' + str(size_in_bytes_max) + '.txt'), 'w') as small_size_file:
        for filepath in list_files:
            small_size_file.write(filepath)
    for filepath in list_files:
        filename = os.path.split(filepath)[1]
        shutil.copy(filepath, os.path.join(dest, filename))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        size_in_bytes_min = 0
        size_in_bytes_max = int(sys.argv[1])
    else:
        for i in range(1, len(sys.argv) - 1):
            size_in_bytes_min = int(sys.argv[i])
            size_in_bytes_max = int(sys.argv[i+1])
            print("Checking min: ", size_in_bytes_min, ", max: ", size_in_bytes_max)
            copy_small_images(train_dir, 'small_files_' + str(size_in_bytes_min) + '_' + str(size_in_bytes_max), size_in_bytes_min, size_in_bytes_max)
