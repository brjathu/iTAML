import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Sampler
from torchvision import datasets, transforms
# from imagenet import ImageNet
from CUB200 import Cub2011
import collections


class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, shuffle):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if(self.shuffle):
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
    

class IncrementalDataset:

    def __init__(
        self,
        dataset_name,
        args,
        random_order=False,
        shuffle=True,
        workers=10,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.
    ):
        self.dataset_name = dataset_name.lower().strip()
        datasets = _get_datasets(dataset_name)
        self.train_transforms = datasets[0].train_transforms  # FIXME handle multiple datasets
        self.common_transforms = datasets[0].common_transforms
        self.args = args
        
        self._setup_data(
            datasets,
            args.data_path,
            random_order=random_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split
        )
        

        self._current_task = 0

        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle

    @property
    def n_tasks(self):
        return len(self.increments)
    
    def get_same_index(self, target, label, mode="train", memory=None):
        label_indices = []
        label_targets = []

        for i in range(len(target)):
            if target[i] in label:
                label_indices.append(i)
                label_targets.append(target[i])

        for_memory = (label_indices.copy(),label_targets.copy())

        if memory is not None:
            memory_indices, memory_targets = memory
            all_indices = memory_indices+label_indices
        else:
            all_indices = label_indices

#         if (mode=="train"):
#             random.shuffle(all_indices)

        return all_indices, for_memory
    
    def get_same_index_test_chunk(self, target, label, mode="test", memory=None):
        label_indices = []
        label_targets = []
        
        np_target = np.array(target, dtype="uint32")
        np_indices = np.array(list(range(len(target))), dtype="uint32")
        
        for t in range(len(label)//self.args.class_per_task):
            task_idx = []
            for class_id in label[t*self.args.class_per_task: (t+1)*self.args.class_per_task]:
                idx = np.where(np_target==class_id)[0]
                task_idx.extend(list(idx.ravel()))
            task_idx = np.array(task_idx, dtype="uint32")
            task_idx.ravel()
            random.shuffle(task_idx)
            label_indices.extend(list(np_indices[task_idx]))
            label_targets.extend(list(np_target[task_idx]))
            
        label_indices = np.array(label_indices, dtype="uint32")
        label_indices.ravel()
        return list(label_indices), label_targets
    

    def new_task(self, memory=None):
        
        print(self._current_task)
        print(self.increments)
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
        
        train_indices, for_memory = self.get_same_index(self.train_dataset.identity, list(range(min_class, max_class)), mode="train", memory=memory)
#         list_a = []
#         for i in range(len(self.train_dataset.targets)):
#             if(i in train_indices):
#                 list_a.append(self.train_dataset.targets[i])
#         print("#############")
#         print(collections.Counter(list_a))
#         print("#############")
        
        test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.identity, list(range(max_class)), mode="test")
        
#         list_a = []
#         for i in range(len(self.test_dataset.targets)):
#             if(i in test_indices):
#                 list_a.append(self.test_dataset.targets[i])
#         print("#############")
#         print(collections.Counter(list_a))
#         print("#############")
        
        
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size,shuffle=False,num_workers=16, sampler=SubsetRandomSampler(train_indices, True))
        self.test_data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=10,shuffle=False,num_workers=16, sampler=SubsetRandomSampler(test_indices, False))

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "increment": self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(train_indices),
            "n_test_data": len(test_indices)
        }

        self._current_task += 1

        return task_info, self.train_data_loader, self.train_data_loader, self.test_data_loader, for_memory
    
 
    def get_custom_loader_idx(self, indexes, mode="train", batch_size=10, shuffle=True):
#         if shuffle:
#             random.shuffle(indexes)
        
        if(mode=="train"):
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, True))
        else: 
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, False))
    
        return data_loader
    
    
    def get_custom_loader_class(self, class_id, mode="train", batch_size=10, shuffle=False):
        
        if(mode=="train"):
            train_indices, for_memory = self.get_same_index(self.train_dataset.targets, class_id, mode="train", memory=None)
#             if shuffle:
#                 random.shuffle(train_indices)
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(train_indices, True))
        else: 
            test_indices, _ = self.get_same_index(self.test_dataset.targets, class_id, mode="test")
#             if shuffle:
#                 random.shuffle(test_indices)
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(test_indices, False))
            
        return data_loader

    def _setup_data(self, datasets, path, random_order=False, seed=1, increment=10, validation_split=0.):
        self.increments = []
        self.class_order = []
        
        trsf_train = transforms.Compose(self.train_transforms)
        trsf_test = transforms.Compose(self.common_transforms)
        
        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:
            if(self.dataset_name=="imagenet"):
                train_dataset = dataset.base_dataset(root=path, split='train', download=False, transform=trsf_train)# ("data", train=True, download=True)
                test_dataset = dataset.base_dataset(root=path, split='val', download=False, transform=trsf_test)
            elif(self.dataset_name=="cub200" or self.dataset_name=="cifar100" or self.dataset_name=="mnist"  or self.dataset_name=="caltech101"):
                train_dataset = dataset.base_dataset(root=path, train=True, download=True, transform=trsf_train)# ("data", train=True, download=True)
                test_dataset = dataset.base_dataset(root=path, train=False, download=True, transform=trsf_test)
            elif(self.dataset_name=="celeb"):
                train_dataset = dataset.base_dataset(root=path, split='train',target_type="identity", download=True, transform=trsf_train)# ("data", train=True, download=True)
                test_dataset = dataset.base_dataset(root=path, split='valid',target_type="identity",  download=True, transform=trsf_test)
                
            order = [i for i in range(self.args.num_class)]
            if random_order:
                random.seed(seed)  # Ensure that following order is determined by seed:
                random.shuffle(order)
            elif dataset.class_order is not None:
                order = dataset.class_order

            self.class_order.append(order)

            if len(datasets) > 1:
                self.increments.append(len(order))
            else:
                self.increments = [increment for _ in range(len(order) // increment)]

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))

def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif dataset_name == "imagenet":
        return iIMAGENET
    elif dataset_name == "cub200":
        return iCUB200
    elif dataset_name == "mnist":
        return iMNIST
    elif dataset_name == "caltech101":
        return iCALTECH101
    elif dataset_name == "celeb":
        return iCELEB
    
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None


class iCIFAR10(DataHandler):
    base_dataset = datasets.cifar.CIFAR10
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]


class iCIFAR100(DataHandler):
    base_dataset = datasets.cifar.CIFAR100
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    
    
class iCALTECH101(DataHandler):
    base_dataset = datasets.Caltech101
    train_transforms = [
        transforms.Resize(136),
        transforms.RandomCrop(128, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    
    common_transforms = [
        transforms.Resize(130),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]

class iCELEB(DataHandler):
    base_dataset = datasets.CelebA
    train_transforms = [
        transforms.Resize(70),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    
    common_transforms = [
        transforms.Resize(70),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]

class iIMAGENET(DataHandler):
    base_dataset = datasets.ImageNet
    train_transforms = [
        transforms.Resize(70),
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    common_transforms = [
        transforms.Resize(70),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]

class iCUB200(DataHandler):
    base_dataset = Cub2011
    train_transforms = [
        transforms.Resize(70),
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        
    ]
    common_transforms = [
        transforms.Resize(70),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ]

    
    
    
class iMNIST(DataHandler):
    base_dataset = datasets.MNIST
    train_transforms = [ transforms.ToTensor()]
    common_transforms = [transforms.ToTensor()]
    
#     def _preprocess_initial_data(self, data):
#         print("$$$$$$$$$$", data.shape)
#         print("$$$$$$$$$$", data.size)
        
#         b, w, h, c = data.shape
#         data = data.reshape(b, 784)
        

class iPermutedMNIST(iMNIST):

    def _preprocess_initial_data(self, data):
        b, w, h, c = data.shape
#         print("$$$$$$$$$$", data.shape)
        data = data.reshape(b, -1, c)

        permutation = np.random.permutation(w * h)

        data = data[:, permutation, :]

        return data.reshape(b, w, h, c)
