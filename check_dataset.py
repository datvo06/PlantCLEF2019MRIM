from __future__ import print_function, unicode_literals, division
import os
from PIL import Image

data_dir = "/video/clef/LifeCLEF/PlantCLEF2019/train/data"
out_list = "list_smallsize.txt"
unopenable_list = "list_unopenable.txt"
out_file = open(out_list, "a")
unopenable_file = open(unopenable_list, "a")
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith('xml'):
            continue
        try:
            pil_im = Image.open(os.path.join(root, filename))
            if os.path.getsize(os.path.join(root, filename)) < 1024:
                out_file.write(os.path.join(root, filename) + '\n')
        except:
            unopenable_file.write(os.path.join(root, filename) + '\n')
out_file.close()
unopenable_file.close()
