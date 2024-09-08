# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
from PIL import Image


"""# Load Dataset"""

class Dataset(data.Dataset):
    def __init__(self, root='./', load_set='train', transform=None):
        self.root = root
        self.transform = transform
        self.load_set = load_set

        self.images = np.load(os.path.join(root, f'images-{load_set}.npy'))
        self.points2d = np.load(os.path.join(root, f'points2d-{load_set}.npy'))
        self.points3d = np.load(os.path.join(root, f'points3d-{load_set}.npy'))

        print(f'Loaded {len(self.images)} images from {root}')
        print(f'Loaded {len(self.points2d)} 2D points')
        print(f'Loaded {len(self.points3d)} 3D points')

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        point2d = self.points2d[index]
        point3d = self.points3d[index]

        if self.transform is not None:
            image = self.transform(image)

        return image[:3], point2d, point3d

    def __len__(self):
        return len(self.images)

