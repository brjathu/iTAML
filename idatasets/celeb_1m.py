import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import numpy as np
import random
from collections import Counter


class MS1M(Dataset):
    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        
        seed = 3
        random.seed(seed)
        np.random.seed(seed)
        
        train_imgs_all = []
        val_imgs_all = []

        
        all_paths = os.listdir(root)
        random.shuffle(all_paths)
        folders = 0

        
        for p in all_paths:
            path_p = root+"/"+ p
            imgs_path = os.listdir(path_p)
            if len(imgs_path)>45:
                folders += 1
                train_imgs = imgs_path[:30]
                val_imgs = imgs_path[30:45]
                for i in train_imgs:
                    full_path = p + "/" + i
                    train_imgs_all.append([full_path, int(p)])

                for i in val_imgs:
                    full_path = p + "/" + i
                    val_imgs_all.append([full_path, int(p)])
            if(folders>=10000):
                break
        train_imgs_all = np.array(train_imgs_all)
        val_imgs_all = np.array(val_imgs_all)

        
#         train_imgs_all = np.load("/raid/brjathu/meta_two/idatasets/train_imgs_all.npy")
#         val_imgs_all = np.load("/raid/brjathu/meta_two/idatasets/val_imgs_all.npy")

        if(self.train):
            data = train_imgs_all
        else:
            data = val_imgs_all
            
        self.data = data[:,0]
        targets_o = data[:,1]
        self.mapped_targets = {}
        c = 0
        for t in np.unique(targets_o):
            self.mapped_targets[t] = c
            c += 1
        
        self.targets = []
        for t in targets_o:
            self.targets.append(int(self.mapped_targets[t]))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        path = os.path.join(self.root, sample)
        target = self.targets[idx]  # Targets start at 1 by default, so shift to 0

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target