import os
import shutil

train_dir = "/video/clef/LifeCLEF/PlantCLEF2019/train/"
if not os.path.exists(os.path.join(train_dir, "store_temp_small_class")):
    os.makedirs(os.path.join(train_dir, "store_temp_small_class"))
data_temp_removed_dir = os.path.join(train_dir, "store_temp_small_class")


def move_small_classes(train_dir, destination, classcount_filepath):
    for row in open(classcount_filepath):
        class_path, class_size = row.split()
        if int(class_size) < 5:
            shutil.move(os.path.join(train_dir, class_path), destination)

if __name__ == '__main__':
    classcount_filepath = os.path.join(train_dir, 'countall.txt')
    move_small_classes(train_dir, data_temp_removed_dir, classcount_filepath)
