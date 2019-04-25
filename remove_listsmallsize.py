import os
import shutil

train_dir = "/video/clef/LifeCLEF/PlantCLEF2019/train/"
if not os.path.exists(os.path.join(train_dir, "store_temp_small_size")):
    os.makedirs(os.path.join(train_dir, "store_temp_small_size"))
data_temp_removed_dir = os.path.join(train_dir, "store_temp_small_class")

def move_small_files(train_dir, destination, list_smallsize_fp):
    with open(list_smallsize_fp) as list_smallsize:
        for row in list_smallsize:
            filepath = row[:-1]
            if not (os.path.exists(filepath)):
                continue
            parent_dir = os.path.basename(filepath)
            if not (os.path.exists(os.path.join(destination, parent_dir))):
                os.makedirs(os.path.join(destination, parent_dir))
            shutil.move(filepath, os.path.join(destination, parent_dir, os.path.split(filepath)[1]))


if __name__ == '__main__':
    move_small_files(train_dir, data_temp_removed_dir, 'list_smallsize.txt')
