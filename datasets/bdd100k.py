
import torch, os
from pathlib import Path
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import numpy as np
from torch import Tensor
import utils, json
from PIL import Image
# import transforms as T
from tqdm import tqdm
from torch import nn


def get_ground_truths(train_img_path_list, anno_data):
    bboxes, total_bboxes = [], []
    labels, total_labels = [], []
    classes = {'bus': 0, 'traffic light': 1, 'traffic sign': 2, 'person': 3, 'bike': 4, 'truck': 5, 'motor': 6,
               'car': 7,
               'train': 8, 'rider': 9, 'drivable area': 10, 'lane': 11}

    for i in tqdm(range(len(train_img_path_list))):
        for j in range(len(anno_data[i]['labels'])):
            if 'box2d' in anno_data[i]['labels'][j]:
                xmin = anno_data[i]['labels'][j]['box2d']['x1']
                ymin = anno_data[i]['labels'][j]['box2d']['y1']
                xmax = anno_data[i]['labels'][j]['box2d']['x2']
                ymax = anno_data[i]['labels'][j]['box2d']['y2']
                bbox = [xmin, ymin, xmax, ymax]
                category = anno_data[i]['labels'][j]['category']
                cls = classes[category]

                bboxes.append(bbox)
                labels.append(cls)

        total_bboxes.append(Tensor(bboxes))
        total_labels.append(Tensor(labels))
        bboxes = []
        labels = []

    return total_bboxes, total_labels


def _load_json(path_list_idx):
    with open(path_list_idx, 'r') as file:
        data = json.load(file)
    return data


# def get_transform(train):
#     transforms = []
#     transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)


class BDD(torch.utils.data.Dataset):
    def __init__(self, img_path, anno_json_path,
                 transforms=None, classes=None):  # total_bboxes_list,total_labels_list,transforms=None):

        super(BDD, self).__init__()
        self.img_path = img_path
        self.anno_data = _load_json(anno_json_path)
        self.total_bboxes_list, self.total_labels_list = get_ground_truths(self.img_path, self.anno_data)
        self.transforms = transforms

        self.classes = {'bus': 0, 'traffic light': 1, 'traffic sign': 2, 'person': 3, 'bike': 4, 'truck': 5, 'motor': 6,
                        'car': 7,
                        'train': 8, 'rider': 9, 'drivable area': 10, 'lane': 11}

        if classes is not None:
            classes_subset = dict()
            for valid_class in classes:
                classes_subset[valid_class] = self.classes[valid_class]

            self.classes = classes_subset

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = Image.open(img_path).convert("RGB")

        labels = self.total_labels_list[idx]
        bboxes = self.total_bboxes_list[idx]
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        img_id = Tensor([idx])
        iscrowd = torch.zeros(len(bboxes, ), dtype=torch.int64)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


if __name__ == '__main__':
    from datasets import load_dataset

    IMG_DIR="/mnt/data/BerkleyBAIR/bdd100k/images/100k/train"
    LABEL_FILE="/mnt/data/BerkleyBAIR/bdd100k/images/100k/bdd100k_labels_images_train.json"
    SIZE=416

    ds = load_dataset(type="bdd",
                      img_dir=IMG_DIR,
                      label_file=LABEL_FILE,
                      img_size=416,
                      batch_size=1,
                      n_cpu=0,
                      shuffle=False,
                      augment=True
                      )

# https://github.com/narumiruna/pytorch-bdd-dataset/blob/master/dataset.py
# import glob
# import os
#
# from torch.utils import data
# from torchvision.datasets.folder import pil_loader
#
# # from utils import load_json
#
#
# class BDDDataset(data.Dataset):
#
#     def __init__(self, root, train=True, transform=None):
#         self.root = root
#         self.train = train
#         self.transform = transform
#         self.samples = None
#
#         self.prepare()
#
#     def prepare(self):
#         self.samples = []
#
#         if self.train:
#             label_paths = glob.glob(
#                 os.path.join(self.root, 'labels/train/*.json'))
#             image_dir = os.path.join(self.root, 'images/100k/train')
#         else:
#             label_paths = glob.glob(
#                 os.path.join(self.root, 'labels/val/*.json'))
#             image_dir = os.path.join(self.root, 'images/100k/val')
#
#         for label_path in label_paths:
#             image_path = os.path.join(
#                 image_dir,
#                 os.path.basename(label_path).replace('.json', '.jpg'))
#
#             if os.path.exists(image_path):
#                 self.samples.append((image_path, label_path))
#             else:
#                 raise FileNotFoundError
#
#     def __getitem__(self, index):
#         # TODO: handle label dict
#
#         image_path, label_path = self.samples[index]
#
#         image = pil_loader(image_path)
#
#
#         # label = load_json(label_path)
#         with open(label_path, 'r') as fp:
#             return json.load(fp)
#
#         if self.transform is not None:
#             image = self.transform(image)
#
#         return image, label
#
#     def __len__(self):
#         return len(self.samples)
#
#
# def main():
#     from torchvision import transforms
#     transform = transforms.Compose(
#         [transforms.Resize(64), transforms.ToTensor()])
#     loader = data.DataLoader(
#         BDDDataset('data/bdd100k', transform=transform),
#         batch_size=2,
#         shuffle=True)
#
#     for i, (x, y) in enumerate(loader):
#         print(x.size())
#         print(y)
#         break