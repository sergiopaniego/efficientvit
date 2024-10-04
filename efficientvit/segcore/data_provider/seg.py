import json
import glob
import os
import h5py


import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools import mask as mask_utils
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from efficientvit.apps.data_provider import DataProvider
from efficientvit.segcore.data_provider.utils import (################3
    Normalize_and_Pad,
    RandomHFlip,
    ResizeLongestSide,
    RandomColorJitter,
    GaussianBlur,
    SEGDistributedSampler, ############
)

__all__ = ["SEGDataProvider"]

# /home/pdi/spaniego/Documentos/gsoc2023-Meiqi_Zhao/src/data/qi_model_dataset/val/episode_1.hdf5

class CARLADataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory with all the hdf5 files containing the episode data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_paths = glob.glob(os.path.join(directory, "*.hdf5"))
        self.file_paths.sort()  # Ensuring data is processed in order
        #self.file_paths = self.file_paths[:1] # DELETE! We take only 1 set of examples for debugging
        
        self.transform = transform

        self.lengths = []
        self.total_length = 0
        self.files = []
        for file_path in self.file_paths:
            file = h5py.File(file_path, 'r')
            self.files.append(file)
            length = file['frame'].shape[0]
            self.lengths.append(length)
            self.total_length += length

    def __len__(self):
        return self.total_length
    

    # one hot encoding for high-level ommand
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # find the file that contains the data for the given index
        file_idx = 0
        while idx >= self.lengths[file_idx]:
            idx -= self.lengths[file_idx]
            file_idx += 1

        file = self.files[file_idx]

        sample = {
            'image': torch.tensor(file['rgb'][idx], dtype=torch.float32).permute(2, 0, 1),
            #'masks': torch.tensor(file['segmentation'][idx], dtype=torch.float32).permute(2, 0, 1),
            'masks': self.convert_rgb_to_class(file['segmentation'][idx]),
            "shape": torch.tensor(torch.tensor(file['rgb'][idx], dtype=torch.float32).permute(2, 0, 1).shape[-2:]),
        }        

        if self.transform:
            sample = self.transform(sample)
       
        return sample

    def convert_rgb_to_class(self, mask):
        """ Convert an RGB mask to a single channel class mask """
        mask = torch.tensor(mask, dtype=torch.float32)
        mask_class = torch.zeros(mask.shape[:2], dtype=torch.long)

        # Assuming you have a mapping from RGB to class values
        # This is a placeholder example; adjust it based on your actual mappings
        rgb_to_class = {
            (128, 64, 128): 0, # "road"
            (244, 35, 232): 1, # "sidewalk"
            (70, 70, 70): 2, # "building"
            (102, 102, 156): 3, # "wall"
            (190, 153, 153): 4, # "fence"
            (153, 153, 153): 5, # "pole"
            (250, 170, 30): 6, # "traffic light"
            (220, 220, 0): 7, # "traffic sign"
            (107, 142, 35): 8, # "vegetation"
            (152, 251, 152): 9, # "terrain"
            (70, 130, 180): 10, # "sky"
            (220, 20, 60): 11, # "person"
            (255, 0, 0): 12, # "rider"
            (0, 0, 142): 13, # "car"
            (0, 0, 70): 14, # "truck"
            (0, 60, 100): 15, # "bus"
            (0, 80, 100): 16, # "train"
            (0, 0, 230): 17, # "motorcycle"
            (119, 11, 32): 18, # "bicycle"
        }

        for rgb, class_value in rgb_to_class.items():
            mask_class[(mask == torch.tensor(rgb, dtype=torch.float32)).all(dim=-1)] = class_value

        return mask_class

    def close(self):
        for file in self.files:
            file.close()


class CARLACityScapesDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory with all the hdf5 files containing the episode data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_paths = glob.glob(os.path.join(directory, "*.hdf5"))
        self.file_paths.sort()  # Ensuring data is processed in order
        #self.file_paths = self.file_paths[:1] # DELETE! We take only 1 set of examples for debugging
        
        self.transform = transform

        self.lengths = []
        self.total_length = 0
        self.files = []
        for file_path in self.file_paths:
            file = h5py.File(file_path, 'r')
            self.files.append(file)
            length = file['frame'].shape[0]
            self.lengths.append(length)
            self.total_length += length

    def __len__(self):
        return self.total_length
    

    # one hot encoding for high-level ommand
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # find the file that contains the data for the given index
        file_idx = 0
        while idx >= self.lengths[file_idx]:
            idx -= self.lengths[file_idx]
            file_idx += 1

        file = self.files[file_idx]

        sample = {
            'image': torch.tensor(file['rgb'][idx], dtype=torch.float32).permute(2, 0, 1),
            #'masks': torch.tensor(file['segmentation'][idx], dtype=torch.float32).permute(2, 0, 1),
            'masks': self.convert_rgb_to_class(file['segmentation'][idx]),
            "shape": torch.tensor(torch.tensor(file['rgb'][idx], dtype=torch.float32).permute(2, 0, 1).shape[-2:]),
        }        

        if self.transform:
            sample = self.transform(sample)
       
        return sample

    def convert_rgb_to_class(self, mask):
        ######
        """ Convert an RGB mask to a single channel class mask """
        mask = torch.tensor(mask, dtype=torch.float32)
        mask_class = torch.zeros(mask.shape[:2], dtype=torch.long)

        # Assuming you have a mapping from RGB to class values
        # This is a placeholder example; adjust it based on your actual mappings
        rgb_to_class = {
            (0, 0, 0): 0, # "unlabeled"
            (110, 190, 160): 1, # "static"
            (170, 120, 50): 2, # "dynamic"
            (81, 0, 81): 3, # "ground"
            (128, 64, 128): 4, # "road"
            (244, 35, 232): 5, # "sidewalk"
            (230, 150, 140): 6, # "rail track"
            (70, 70, 70): 7, # "building"
            (102, 102, 156): 8, # "wall"
            (190, 153, 153): 9, # "fence"
            (180, 165, 180): 10, # "guard rail"
            (150, 100, 100): 11, # "bridge"
            (153, 153, 153): 12, # "pole"
            (250, 170, 30): 13, # "traffic light"
            (220, 220, 0): 14, # "traffic sign"
            (107, 142, 35): 15, # "vegetation"
            (152, 251, 152): 16, # "terrain"
            (70, 130, 180): 17, # "sky"
            (220, 20, 60): 18, # "person"
            (255, 0, 0): 19, # "rider"
            (0, 0, 142): 20, # "car"
            (0, 0, 70): 21, # "truck"
            (0, 60, 100): 22, # "bus"
            (0, 80, 100): 23, # "train"
            (0, 0, 230): 24, # "motorcycle"
            (119, 11, 32): 25, # "bicycle"
            (55, 90, 80): 26, # "other"
            (45, 60, 150): 27, # "water"
            (157, 234, 50): 28, # "road line"
        }

        for rgb, class_value in rgb_to_class.items():
            mask_class[(mask == torch.tensor(rgb, dtype=torch.float32)).all(dim=-1)] = class_value

        return mask_class

    def close(self):
        for file in self.files:
            file.close()

            
class OnlineDataset(Dataset):
    def __init__(self, root, train=True, num_masks=64, transform=None):
        self.root = root
        self.train = train
        self.num_masks = num_masks
        self.transform = transform

        self.data = open(f"{self.root}/sa_images_ids.txt", "r").read().splitlines()

        if self.train:
            self.data = self.data[: int(len(self.data) * 0.99)]
        else:
            self.data = self.data[int(len(self.data) * 0.99) :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Note: We provide the simplest data organization here. You can modify the code according to your data organization.
        """

        index = int(self.data[idx])

        image_path = f"{self.root}/images/sa_{index}.jpg"
        image = io.imread(image_path)

        json_path = f"{self.root}/masks/sa_{index}.json"
        annotations = json.load(open(json_path))["annotations"]

        if self.train:
            if len(annotations) > self.num_masks:
                r = np.random.choice(len(annotations), size=self.num_masks, replace=False)
            else:
                repeat, residue = self.num_masks // len(annotations), self.num_masks % len(annotations)
                r = np.random.choice(len(annotations), size=residue, replace=False)
                r = np.concatenate([np.arange(len(annotations)) for _ in range(repeat)] + [r], axis=0)

        else:
            if len(annotations) > self.num_masks:
                r = np.arange(self.num_masks)
            else:
                repeat, residue = self.num_masks // len(annotations), self.num_masks % len(annotations)
                r = np.arange(residue)
                r = np.concatenate([np.arange(len(annotations)) for _ in range(repeat)] + [r], axis=0)

        masks = np.stack([mask_utils.decode(annotations[i]["segmentation"]) for i in r])
        #points = np.stack([annotations[i]["point_coords"][0] for i in r])
        #bboxs = np.stack([annotations[i]["bbox"] for i in r])

        image = torch.tensor(image, dtype=torch.float32)
        image = torch.transpose(torch.transpose(image, 1, 2), 0, 1)
        masks = torch.tensor(masks, dtype=torch.float32)
        #points = torch.tensor(points, dtype=torch.float32)
        #bboxs = torch.tensor(bboxs, dtype=torch.float32)

        sample = {
            "image": image,
            "masks": masks,
            #"points": points,
            #"bboxs": bboxs,
            "shape": torch.tensor(image.shape[-2:]),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class SEGDataProvider(DataProvider):
    name = "seg"

    def __init__(
        self,
        root: str,
        sub_epochs_per_epoch: int,
        #num_masks: int,
        train_batch_size: int,
        test_batch_size: int,
        valid_size: int or float or None = None,
        n_worker=8,
        image_size: int = 1024,
        num_replicas: int or None = None,
        rank: int or None = None,
        train_ratio: float or None = None,
        drop_last: bool = False,
    ):
        self.root = root
        #self.num_masks = num_masks
        self.sub_epochs_per_epoch = sub_epochs_per_epoch

        super().__init__(
            train_batch_size,
            test_batch_size,
            valid_size,
            n_worker,
            image_size,
            num_replicas,
            rank,
            train_ratio,
            drop_last,
        )

    def build_train_transform(self):
        train_transforms = [
            RandomHFlip(),
            RandomColorJitter(brightness=0.3, contrast=0.3),
            GaussianBlur(sigma=(0.1, 2.0)),
            ResizeLongestSide(target_length=self.image_size[0]),
            Normalize_and_Pad(target_length=self.image_size[0]),
        ]

        return transforms.Compose(train_transforms)

    def build_valid_transform(self):
        valid_transforms = [
            ResizeLongestSide(target_length=self.image_size[0]),
            Normalize_and_Pad(target_length=self.image_size[0]),
        ]

        return transforms.Compose(valid_transforms)

    def build_datasets(self) -> tuple[any, any, any]:
        train_transform = self.build_train_transform()
        valid_transform = self.build_valid_transform()

        #train_dataset = OnlineDataset(root=self.root, train=True, num_masks=self.num_masks, transform=train_transform)
        #val_dataset = OnlineDataset(root=self.root, train=False, num_masks=2, transform=valid_transform)

#####
        #train_dataset = CARLADataset(directory=f"{self.root}train", transform=train_transform)
        #val_dataset = CARLADataset(directory=f"{self.root}val", transform=valid_transform)

        train_dataset = CARLACityScapesDataset(directory=f"{self.root}train", transform=train_transform)
        val_dataset = CARLACityScapesDataset(directory=f"{self.root}val", transform=valid_transform)

        test_dataset = None

        return train_dataset, val_dataset, test_dataset

    def build_dataloader(self, dataset: any or None, batch_size: int, n_worker: int, drop_last: bool, train: bool):
        if dataset is None:
            return None
        if train:
            #sampler = SEGDistributedSampler(dataset, sub_epochs_per_epoch=self.sub_epochs_per_epoch, num_replicas=1) #################################
            sampler = SEGDistributedSampler(dataset, num_replicas=1, rank=0,sub_epochs_per_epoch=self.sub_epochs_per_epoch)

            dataloader = DataLoader(dataset, batch_size, sampler=sampler, drop_last=True, num_workers=n_worker)
            #dataloader = DataLoader(dataset, batch_size, drop_last=True, num_workers=n_worker, shuffle=True)
            return dataloader
        else:
            sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
            dataloader = DataLoader(dataset, batch_size, sampler=sampler, drop_last=False, num_workers=n_worker)
            #dataloader = DataLoader(dataset, batch_size, drop_last=False, num_workers=n_worker, shuffle=True)
            return dataloader

    def set_epoch_and_sub_epoch(self, epoch: int, sub_epoch: int) -> None:
        if isinstance(self.train.sampler, SEGDistributedSampler):
            self.train.sampler.set_epoch_and_sub_epoch(epoch, sub_epoch)
