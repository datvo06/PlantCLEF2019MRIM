from __future__ import print_function, division, unicode_literals
import os
import numpy as np
from matplotlib import pyplot as plt


list_count = []
with open('/video/clef/LifeCLEF/PlantCLEF2019/train/multiples.txt') as countfile:
    for line in countfile:
        list_count.append(int(line.split()[1]))
    list_count = [x for x in list_count if x >= 2]
    list_count = np.array(list_count)
    print(len([x for x in list_count if x == 2]))
    list_set = set(list_count)
    print(len(list_set))
    print("Total dup: ", sum(list_count))
    print("Mean dup: ", sum(list_count)/len(list_count))
    print("Median dup: ", np.median(list_count))
    print("Max list set: ", max(list_set))
    plt.hist(list_count, bins=325)
    plt.title('Histogram of duplicated samples')
    plt.savefig('sample_dup.jpg')
