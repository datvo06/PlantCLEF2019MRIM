from __future__ import print_function, division, unicode_literals
import os
import numpy as np
from matplotlib import pyplot as plt


list_count = []
with open('/video/clef/LifeCLEF/PlantCLEF2019/train/countall.txt') as countfile:
    for line in countfile:
        list_count.append(int(line.split()[1]))
    list_count = [x for x in list_count if x >= 5]
    list_count = np.array(list_count)
    list_set = set(list_count)
    print(len(list_set))
    print("Total len: ", sum(list_count))
    print("Mean len: ", sum(list_count)/len(list_count))
    print("Median len: ", np.median(list_count))
    print("Max list set: ", max(list_set))
    plt.hist(list_count, bins=325)
    plt.title('Number of sample per class distribution clipped from 5')
    plt.savefig('sample_hist_atleast5.jpg')
