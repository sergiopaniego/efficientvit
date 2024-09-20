# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import math
import os
import pathlib

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

from efficientvit.apps.utils import AverageMeter
from efficientvit.models.utils import resize
from efficientvit.seg_model_zoo import create_seg_model


class Resize(object):
    def __init__(
        self,
        crop_size,
        interpolation = cv2.INTER_CUBIC,
    ):
        self.crop_size = crop_size
        self.interpolation = interpolation

    def __call__(self, feed_dict):
        if self.crop_size is None or self.interpolation is None:
            return feed_dict

        image, target = feed_dict["data"], feed_dict["label"]
        height, width = self.crop_size

        print(image.shape)
        

        h, w, _ = image.shape
        if width != w or height != h:
            image = cv2.resize(
                image,
                dsize=(width, height),
                interpolation=self.interpolation,
            )
        return {
            "data": image,
            "label": target,
        }


class ToTensor(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, feed_dict):
        image, mask = feed_dict["data"], feed_dict["label"]
        image = image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = torch.as_tensor(image, dtype=torch.float32).div(255.0)
        mask = torch.as_tensor(mask, dtype=torch.int64)
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return {
            "data": image,
            "label": mask,
        }


class SegIOU:
    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor):
        outputs = (outputs + 1) * (targets != self.ignore_index)
        targets = (targets + 1) * (targets != self.ignore_index)
        intersections = outputs * (outputs == targets)

        outputs = torch.histc(
            outputs,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        targets = torch.histc(
            targets,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        intersections = torch.histc(
            intersections,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        unions = outputs + targets - intersections

        return {
            "i": intersections,
            "u": unions,
        }


class CityscapesDataset(Dataset):
    classes = (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )
    class_colors = (
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    )
    label_map = np.array(
        (
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            0,  # road 7
            1,  # sidewalk 8
            -1,
            -1,
            2,  # building 11
            3,  # wall 12
            4,  # fence 13
            -1,
            -1,
            -1,
            5,  # pole 17
            -1,
            6,  # traffic light 19
            7,  # traffic sign 20
            8,  # vegetation 21
            9,  # terrain 22
            10,  # sky 23
            11,  # person 24
            12,  # rider 25
            13,  # car 26
            14,  # truck 27
            15,  # bus 28
            -1,
            -1,
            16,  # train 31
            17,  # motorcycle 32
            18,  # bicycle 33
        )
    )

    def __init__(self, data_dir: str, crop_size= None):
        super().__init__()

        # load samples
        samples = []
        for dirpath, _, fnames in os.walk(data_dir):
            for fname in sorted(fnames):
                suffix = pathlib.Path(fname).suffix
                if suffix not in [".png"]:
                    continue
                image_path = os.path.join(dirpath, fname)
                mask_path = image_path.replace("/leftImg8bit/", "/gtFine/").replace(
                    "_leftImg8bit.", "_gtFine_labelIds."
                )
                #print('image_path',image_path)
                #print('mask_path', mask_path)
                if not mask_path.endswith(".png"):
                    mask_path = ".".join([*mask_path.split(".")[:-1], "png"])
                samples.append((image_path, mask_path))
        self.samples = samples

        # build transform
        self.transform = transforms.Compose(
            [
                Resize(crop_size),
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        #mask = self.label_map[mask]

        feed_dict = {
            "data": image,
            "label": mask,
        }
        feed_dict = self.transform(feed_dict)
        return {
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path,
            **feed_dict,
        }

class CityscapesDatasetCarla(Dataset):
    classes = (
        "unlabeled",
        "static",
        "dynamic",
        "ground",

        "road",
        "sidewalk",

        "rail track",

        "building",
        "wall",
        "fence",

        "guard rail",
        "bridge",

        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "other",
        "water",
        "road line"
    )
    class_colors = (
        (0, 0, 0), # Unlabeled
        (110, 190, 160), # Static
        (170, 120, 50), # Dynamic
        (81, 0, 81), # Ground

        (128, 64, 128), # Roads
        (244, 35, 232), # SideWalks

        (230, 150, 140), # RailTrack

        (70, 70, 70), # Building
        (102, 102, 156), # Wall
        (190, 153, 153), # Fence

        (180, 165, 180), # GuardRail
        (150, 100, 100), # Bridge

        (153, 153, 153), # Pole
        (250, 170, 30), # TrafficLight
        (220, 220, 0), # TrafficSign
        (107, 142, 35), # Vegetation
        (152, 251, 152), # Terrain
        (70, 130, 180), # Sky
        (220, 20, 60), # Pedestrian
        (255, 0, 0), # Rider
        (0, 0, 142), # Car
        (0, 0, 70), # Truck
        (0, 60, 100), # Bus
        (0, 80, 100), # Train
        (0, 0, 230), # Motorcycle
        (119, 11, 32), # Bicycle

        (55, 90, 80), ###### Other
        (45, 60, 150), ##### Water
        (157, 234, 50), ##### RoadLine
    )
    label_map = np.array(
        (
            0, ## unlabeled 0
            -1,
            -1,
            -1,
            1, ## static 4
            2, ## dynamic 5
            3, ## ground 6
            4,  # road 7
            5,  # sidewalk 8
            -1, 
            6, ## rail track 9
            7,  # building 11
            8,  # wall 12
            9,  # fence 13
            10, ## guard rail 14
            11, ## bridge 15
            -1,
            12,  # pole 17
            -1,
            13,  # traffic light 19
            14,  # traffic sign 20
            15,  # vegetation 21
            16,  # terrain 22
            17,  # sky 23
            18,  # person 24
            19,  # rider 25
            20,  # car 26
            21,  # truck 27
            22,  # bus 28
            -1,
            -1,
            23,  # train 31
            24,  # motorcycle 32
            25,  # bicycle 33

            26,  ## other
            27,  ## water
            28,  ## road line
        )
    )

    def __init__(self, data_dir: str, crop_size= None):
        super().__init__()

        # load samples
        samples = []
        for dirpath, _, fnames in os.walk(data_dir):
            for fname in sorted(fnames):
                suffix = pathlib.Path(fname).suffix
                if suffix not in [".png"]:
                    continue
                image_path = os.path.join(dirpath, fname)
                mask_path = image_path.replace("/leftImg8bit/", "/gtFine/").replace(
                    "_leftImg8bit.", "_gtFine_labelIds."
                )
                #print('image_path',image_path)
                #print('mask_path', mask_path)
                if not mask_path.endswith(".png"):
                    mask_path = ".".join([*mask_path.split(".")[:-1], "png"])
                samples.append((image_path, mask_path))
        self.samples = samples

        # build transform
        self.transform = transforms.Compose(
            [
                Resize(crop_size),
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        #mask = self.label_map[mask]

        feed_dict = {
            "data": image,
            "label": mask,
        }
        feed_dict = self.transform(feed_dict)
        return {
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path,
            **feed_dict,
        }

class ADE20KDataset(Dataset):
    classes = (
        "wall",
        "building",
        "sky",
        "floor",
        "tree",
        "ceiling",
        "road",
        "bed",
        "windowpane",
        "grass",
        "cabinet",
        "sidewalk",
        "person",
        "earth",
        "door",
        "table",
        "mountain",
        "plant",
        "curtain",
        "chair",
        "car",
        "water",
        "painting",
        "sofa",
        "shelf",
        "house",
        "sea",
        "mirror",
        "rug",
        "field",
        "armchair",
        "seat",
        "fence",
        "desk",
        "rock",
        "wardrobe",
        "lamp",
        "bathtub",
        "railing",
        "cushion",
        "base",
        "box",
        "column",
        "signboard",
        "chest of drawers",
        "counter",
        "sand",
        "sink",
        "skyscraper",
        "fireplace",
        "refrigerator",
        "grandstand",
        "path",
        "stairs",
        "runway",
        "case",
        "pool table",
        "pillow",
        "screen door",
        "stairway",
        "river",
        "bridge",
        "bookcase",
        "blind",
        "coffee table",
        "toilet",
        "flower",
        "book",
        "hill",
        "bench",
        "countertop",
        "stove",
        "palm",
        "kitchen island",
        "computer",
        "swivel chair",
        "boat",
        "bar",
        "arcade machine",
        "hovel",
        "bus",
        "towel",
        "light",
        "truck",
        "tower",
        "chandelier",
        "awning",
        "streetlight",
        "booth",
        "television receiver",
        "airplane",
        "dirt track",
        "apparel",
        "pole",
        "land",
        "bannister",
        "escalator",
        "ottoman",
        "bottle",
        "buffet",
        "poster",
        "stage",
        "van",
        "ship",
        "fountain",
        "conveyer belt",
        "canopy",
        "washer",
        "plaything",
        "swimming pool",
        "stool",
        "barrel",
        "basket",
        "waterfall",
        "tent",
        "bag",
        "minibike",
        "cradle",
        "oven",
        "ball",
        "food",
        "step",
        "tank",
        "trade name",
        "microwave",
        "pot",
        "animal",
        "bicycle",
        "lake",
        "dishwasher",
        "screen",
        "blanket",
        "sculpture",
        "hood",
        "sconce",
        "vase",
        "traffic light",
        "tray",
        "ashcan",
        "fan",
        "pier",
        "crt screen",
        "plate",
        "monitor",
        "bulletin board",
        "shower",
        "radiator",
        "glass",
        "clock",
        "flag",
    )
    class_colors = (
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    )

    def __init__(self, data_dir: str, crop_size=512):
        super().__init__()

        self.crop_size = crop_size
        # load samples
        samples = []
        for dirpath, _, fnames in os.walk(data_dir):
            for fname in sorted(fnames):
                suffix = pathlib.Path(fname).suffix
                if suffix not in [".jpg"]:
                    continue
                image_path = os.path.join(dirpath, fname)
                mask_path = image_path.replace("/images/", "/annotations/")
                if not mask_path.endswith(".png"):
                    mask_path = ".".join([*mask_path.split(".")[:-1], "png"])
                samples.append((image_path, mask_path))
        self.samples = samples

        self.transform = transforms.Compose(
            [
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.int64) - 1

        h, w = image.shape[:2]
        if h < w:
            th = self.crop_size
            tw = math.ceil(w / h * th / 32) * 32
        else:
            tw = self.crop_size
            th = math.ceil(h / w * tw / 32) * 32
        if th != h or tw != w:
            image = cv2.resize(
                image,
                dsize=(tw, th),
                interpolation=cv2.INTER_CUBIC,
            )

        feed_dict = {
            "data": image,
            "label": mask,
        }
        feed_dict = self.transform(feed_dict)
        return {
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path,
            **feed_dict,
        }


def get_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    colors: tuple or list,
    opacity=0.5,
):
    image_shape = image.shape[:2]
    mask_shape = mask.shape
    if image_shape != mask_shape:
        mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    seg_mask = np.zeros_like(image, dtype=np.uint8)
    for k, color in enumerate(colors):
        print(k, color)
        seg_mask[mask == k, :] = color
    canvas = seg_mask #* opacity + image * (1 - opacity)
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/dataset/cityscapes/leftImg8bit/val")
    parser.add_argument("--dataset", type=str, default="cityscapes", choices=["cityscapes", "ade20k"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=1)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.batch_size = args.batch_size * max(len(device_list), 1)

    if args.dataset == "cityscapes":
        print('CROP SIZE', (args.crop_size, args.crop_size * 2))
        dataset = CityscapesDataset(args.path, (args.crop_size, args.crop_size * 2))
    elif args.dataset == "ade20k":
        dataset = ADE20KDataset(args.path, crop_size=args.crop_size)
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    model = create_seg_model(args.model, args.dataset, weight_url=args.weight_url)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
    interaction = AverageMeter(is_distributed=False)
    union = AverageMeter(is_distributed=False)
    iou = SegIOU(len(dataset.classes))
    print(args.path)
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {args.model} on {args.dataset}") as t:
            for feed_dict in data_loader:
                images, mask = feed_dict["data"].cuda(), feed_dict["label"].cuda()
                print('data input shape', images.shape)
                # compute output
                output = model(images)
                print('output', output)
                print('output.shape', output.shape)
                print('mask.shape', mask.shape)
                # resize the output to match the shape of the mask
                if output.shape[-2:] != mask.shape[-2:]:
                    print('RESIZE!', mask.shape[-2:])
                    print('POSSIBLE RESIZE!', mask.shape[1:3])
                    #output = resize(output, size=mask.shape[-2:])
                    output = resize(output, size=mask.shape[1:3])

                print('output.shape', output.shape)
                print('mask.shape', mask.shape)
                output = torch.argmax(output, dim=1)
                '''
                stats = iou(output, mask)
                interaction.update(stats["i"])
                union.update(stats["u"])

                t.set_postfix(
                    {
                        "mIOU": (interaction.sum / union.sum).cpu().mean().item() * 100,
                        "image_size": list(images.shape[-2:]),
                    }
                )
                t.update()
                '''

                if args.save_path is not None:
                    with open(os.path.join(args.save_path, "summary.txt"), "a") as fout:
                        for i, (idx, image_path) in enumerate(zip(feed_dict["index"], feed_dict["image_path"])):
                            pred = output[i].cpu().numpy()
                            print(image_path)
                            raw_image = np.array(Image.open(image_path).convert("RGB"))
                            canvas = get_canvas(raw_image, pred, dataset.class_colors)
                            #canvas = Image.fromarray(canvas)
                            canvas = Image.fromarray(canvas).save(os.path.join(args.save_path, f"{idx}.png"))
                            #canvas.save(os.path.join(args.save_path, f"{idx}.png"))
                            print(idx)
                            print(image_path)
                            #fout.write(f"{idx}:\t{image_path}\n")

    print(f"mIoU = {(interaction.sum / union.sum).cpu().mean().item() * 100:.3f}")


if __name__ == "__main__":
    main()
