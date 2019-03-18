from __future__ import print_function, unicode_literals, division
import os
from PIL import Image

data_dir = "/video/clef/LifeCLEF/PlantCLEF2019/train/data"
out_list = "list_corrupt.txt"
open(out_list, "a")
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        try:
            im = Image.load(filename)
            im.verify()
            im.close()
            im = Image.load(filename)
            im.transpose(Image.FLIP_LEFT_RIGHT)
            im.close()
        except:
                out_list.write(os.path.join(root, filename) + '\n')
out_list.close()
