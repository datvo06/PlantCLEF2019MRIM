from __future__ import print_function, unicode_literals, division
import os
import pickle as pkl
import sys


def get_all_images_path(data_path, list_classes):
    class_id_tuple_list = list([(class_id, target_id) for class_id, target_id in enumerate(list_classes)])
    mapping_dict = dict(class_id_tuple_list)
    class_images = [[] for i in range(len(list_classes))]
    for class_id, class_name in enumerate(list_classes):
        for root_path, dirpath, filepaths in os.walk(os.path.join(data_path, class_name)):
            for filepath in filepaths:
                if not filepath.endswith('xml'):
                    class_images[class_id].append(os.path.join(root_path, filepath))
    return mapping_dict, class_images


if __name__ == '__main__':
    list_classes = list(open('/video/clef/LifeCLEF/PlantCLEF2017/web/list.txt', 'r'))
    list_classes = [classname.split('/')[1][:-1] for classname in list_classes]
    mapping_dict, class_images = get_all_images_path('/video/clef/LifeCLEF/PlantCLEF2017/web/data', list_classes)
    pkl.dump((mapping_dict, class_images), open('train_set_2017_web.pkl', 'wb'))
