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
    #RandomHFlip,
    ResizeLongestSide,
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
            'masks': torch.tensor(file['segmentation'][idx], dtype=torch.float32).permute(2, 0, 1),
            "shape": torch.tensor(torch.tensor(file['rgb'][idx], dtype=torch.float32).permute(2, 0, 1).shape[-2:]),
        }

        if self.transform:
            sample = self.transform(sample)


        sample['image'] = sample['image'] / 255.0
        sample['masks'] = sample['masks'] / 255.0

       
        return sample

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
            #RandomHFlip(),
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

        train_dataset = CARLADataset(directory=f"{self.root}train", transform=train_transform)
        val_dataset = CARLADataset(directory=f"{self.root}val", transform=valid_transform)

        test_dataset = None

        return train_dataset, val_dataset, test_dataset

    def build_dataloader(self, dataset: any or None, batch_size: int, n_worker: int, drop_last: bool, train: bool):
        if dataset is None:
            return None
        if train:
            #sampler = SEGDistributedSampler(dataset, sub_epochs_per_epoch=self.sub_epochs_per_epoch) #################################
            #dataloader = DataLoader(dataset, batch_size, sampler=sampler, drop_last=True, num_workers=n_worker)
            dataloader = DataLoader(dataset, batch_size, drop_last=True, num_workers=n_worker)
            return dataloader
        else:
            #sampler = DistributedSampler(dataset, shuffle=False)
            #dataloader = DataLoader(dataset, batch_size, sampler=sampler, drop_last=False, num_workers=n_worker)
            dataloader = DataLoader(dataset, batch_size, drop_last=False, num_workers=n_worker)
            return dataloader

    def set_epoch_and_sub_epoch(self, epoch: int, sub_epoch: int) -> None:
        if isinstance(self.train.sampler, SEGDistributedSampler):
            self.train.sampler.set_epoch_and_sub_epoch(epoch, sub_epoch)
