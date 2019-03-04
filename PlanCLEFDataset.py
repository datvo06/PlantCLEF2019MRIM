from __future__ import print_function, division
import torch
import os
from skimage import io as skio, transform as sktransform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class PlanCLEF2017Dataset(Dataset):
    '''Dataset to load plan clef 2017'''

    def __init__(self, root_eol, transform=None):
        with open(os.path.join(root_eol, 'keylist.txt'), 'r') as list_file:
            self.lists_images = [line for line in list_file]
        with open(os.path.join(root_eol, 'lists.txt'), 'r') as list_classes:
            self.lists_classes = [line for line in list_classes]
        self.lists_classes = [int(classname[5:])
                              for classname in self.lists_classes]
        self.root_dir = root_eol
        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.lists_images[idx])
        classId = int(os.path.dirname(self.lists_images[idx])[5:])
        image = skio.imread(img_name)
        sample = {'image': image, 'id': self.lists_classes.index(classId)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Rescale(object):
    '''Rescale the image in a sample to a given size
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched to
        output_size keeping the aspect ratio the same
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

    def __call__(self, sample):
        image, classid = sample['image'], sample['id']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image = sktransform.resize(image, (new_h, new_w))
        return {'image': image, 'id': classid}


class RandomCrop(object):
    '''Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size.
        If int, square crop is made
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, classid = sample['image'], sample['id']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top+new_h,
                      left: left+new_w]
        return {'image': image, 'id': classid}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, classid = sample['image'], sample['classid']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'classid': torch.from_numpy(classid)}


transformed_dataset = PlanCLEF2017Dataset(
    root_eol='video/clef/LifeCLEF/PlantCLEF2017/eol',
    transform=transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor()
    ]))


dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
