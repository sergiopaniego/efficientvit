import random
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class SEGDistributedSampler(DistributedSampler):
    """
    Modified from https://github.com/pytorch/pytorch/blob/97261be0a8f09bed9ab95d0cee82e75eebd249c3/torch/utils/data/distributed.py.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        sub_epochs_per_epoch: int = 1,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.sub_epoch = 0
        self.sub_epochs_per_epoch = sub_epochs_per_epoch
        self.set_sub_num_samples()

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        indices = indices[(self.sub_epoch % self.sub_epochs_per_epoch) :: self.sub_epochs_per_epoch]

        return iter(indices)

    def __len__(self) -> int:
        return self.sub_num_samples

    def set_sub_num_samples(self) -> int:
        self.sub_num_samples = self.num_samples // self.sub_epochs_per_epoch
        if self.sub_num_samples % self.sub_epochs_per_epoch > self.sub_epoch:
            self.sub_num_samples += 1

    def set_epoch_and_sub_epoch(self, epoch: int, sub_epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            sub_epoch (int): Sub epoch number.
        """
        self.epoch = epoch
        self.sub_epoch = sub_epoch
        self.set_sub_num_samples()

'''
class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, masks, points, bboxs, shape = (
            sample["image"],
            sample["masks"],
            sample["points"],
            sample["bboxs"],
            sample["shape"],
        )

        if random.random() >= self.prob:
            image = torch.flip(image, dims=[2])
            masks = torch.flip(masks, dims=[2])
            points = deepcopy(points).to(torch.float)
            bboxs = deepcopy(bboxs).to(torch.float)
            points[:, 0] = shape[-1] - points[:, 0]
            bboxs[:, 0] = shape[-1] - bboxs[:, 2] - bboxs[:, 0]

        return {"image": image, "masks": masks, "points": points, "bboxs": bboxs, "shape": shape}
'''

class ResizeLongestSide(object):
    """
    Modified from https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/utils/transforms.py.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        scale = long_side_length / max(oldh, oldw)
        newh, neww = int(oldh * scale), int(oldw * scale)
        return (newh, neww)

    def apply_image(self, image: torch.Tensor, original_size: Tuple[int, ...]) -> torch.Tensor:
        target_size = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        return F.interpolate(image.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False, antialias=True).squeeze(0)

    def apply_mask(self, mask: torch.Tensor, original_size: Tuple[int, ...]) -> torch.Tensor:
        target_size = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        mask = mask.unsqueeze(0).unsqueeze(0).float()  # Asegurarse de que tenga forma [1, 1, H, W]
        resized_mask = F.interpolate(mask, size=target_size, mode="nearest").squeeze(0).squeeze(0).long()
        return resized_mask

    def __call__(self, sample: dict) -> dict:
        image = sample['image']
        masks = sample['masks']
        original_size = image.shape[1:]  # Assumes image is [C, H, W]

        sample['image'] = self.apply_image(image, original_size)        
        sample['masks'] = self.apply_mask(masks, original_size)
        sample['shape'] = torch.tensor(sample['image'].shape[-2:])

        return sample


class Normalize_and_Pad(object):
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length
        self.transform = transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    def __call__(self, sample):
        image, masks, shape = (
            sample["image"],
            sample["masks"],
            sample["shape"],
        )

        h, w = image.shape[-2:]
        image = self.transform(image)

        padh = self.target_length - h
        padw = self.target_length - w

        image = F.pad(image.unsqueeze(0), (0, padw, 0, padh), value=0).squeeze(0)
        #masks = F.pad(masks.unsqueeze(1), (0, padw, 0, padh), value=0).squeeze(1)   
        masks = F.pad(masks.unsqueeze(0).float(), (0, padw, 0, padh), value=0).squeeze(0).long()


        return {"image": image, "masks": masks, "shape": shape}
        #return {"image": image, "masks": masks}
