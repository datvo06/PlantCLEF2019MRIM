import os
import csv
from bs4 import BeautifulSoup

# First get all the folders
# Get all xml in the folder
# Get the tags
# write all to excel file
if __name__ == '__main__':
    base_path = '/video/clef/LifeCLEF/PlantCLEF2019/train/store_temp_small_class/'
    dirlist = filter(
            lambda dirname: os.path.isdir(os.path.join(base_path, dirname)),
            os.listdir(base_path))
    with open('list_class_small_names.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Species Name'])
        for directory in dirlist:
            # Get dir name
            dirname = os.path.split(directory)[1]
            if dirname.endswith("jpg"):
                continue
            dir_path = os.path.join(base_path, dirname)
            # Get all xml
            filelist = filter(
                    lambda filename: filename.endswith('xml'),
                    os.listdir(dir_path))
            list_unique_names = []
            for fname in filelist:
                fpath = os.path.join(dir_path, fname)
                with open(fpath, "r") as xmlfile:
                    xml_text = xmlfile.read()
                    soup = BeautifulSoup(xml_text, 'lxml')
                    species = soup.find(lambda tag: tag.name.lower() == "species")
                    list_unique_names.append(species.text)
            list_unique_names = list(set(list_unique_names))
            list_unique_names = [dirname] + list_unique_names
            writer.writerow(list_unique_names)
