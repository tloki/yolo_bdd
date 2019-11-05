# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2019 Western Digital Corporation or its affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
from datasets.image import ImageFolder
from datasets.coco_old import CocoDetectionBoundingBox
from datasets.caltech import CaltechPedDataset
from torch.utils.data import DataLoader
from datasets.bdd100k import BDD


def load_dataset(type_: str, img_dir, label_file, img_size, batch_size, n_cpu, shuffle, augment, need_padding=False, **kwargs):
    type_ = type_.lower()

    if type_ == "image_folder":
        _dataset = ImageFolder(img_dir, img_size=img_size)
        _collate_fn = None
    elif type_ == "coco" or type_ == "bdd100k":
        _transform = 'random' if augment else 'default'
        _dataset = CocoDetectionBoundingBox(img_dir, label_file, img_size=img_size, transform=_transform,
                                            need_padding=need_padding)
        _collate_fn = collate_img_label_fn
    elif type_ == "caltech":
        _dataset = CaltechPedDataset(img_dir, img_size, **kwargs)
        _collate_fn = collate_img_label_fn
    # elif type_ == "bdd" or type_ == "bdd100k" or type_ == "berkley":
    #     from datasets.bdd_data.bdd2coco import bdd2coco_detection
    #     import json
    #     attr_dict = dict()
    #     attr_dict["categories"] = [
    #         {"supercategory": "none", "id": 0, "name": "person"},
    #         {"supercategory": "none", "id": 1, "name": "rider"},
    #         {"supercategory": "none", "id": 2, "name": "car"},
    #         {"supercategory": "none", "id": 3, "name": "bus"},
    #         {"supercategory": "none", "id": 4, "name": "truck"},
    #         {"supercategory": "none", "id": 5, "name": "bike"},
    #         {"supercategory": "none", "id": 6, "name": "motor"},
    #         {"supercategory": "none", "id": 7, "name": "traffic light"},
    #         {"supercategory": "none", "id": 8, "name": "traffic sign"},
    #         {"supercategory": "none", "id": 9, "name": "train"}
    #     ]
    #     attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
    #
    #     with open(label_file) as f:
    #         train_labels = json.load(f)
    #
    #     bdd2coco_detection(attr_id_dict, train_labels, label_file + ".coco_format.json")
    #
    #     _transform = 'random' if augment else 'default'
    #     _dataset = CocoDetectionBoundingBox(img_dir, label_file, img_size=img_size, transform=_transform)
    #     _collate_fn = collate_img_label_fn

    else:
        raise TypeError("dataset types can only be 'image_folder', 'coco' or 'caltech'.")
    if _collate_fn is not None:
        _dataloader = DataLoader(_dataset, batch_size, shuffle, num_workers=n_cpu, collate_fn=_collate_fn)
    else:
        _dataloader = DataLoader(_dataset, batch_size, shuffle, num_workers=n_cpu)
    return _dataloader


def collate_img_label_fn(sample):
    images = []
    labels = []
    lengths = []
    labels_with_tail = []
    max_num_obj = 0
    return_more = False
    scales = []
    paddings = []
    for e in sample:
        if not return_more and len(e) > 3:
            return_more = True

        if return_more:
            scales.append(e[3])
            paddings.append(e[4])

        image, label, length = e[0:3]
        images.append(image)
        labels.append(label)
        lengths.append(length)
        max_num_obj = max(max_num_obj, length)
    for label in labels:
        num_obj = label.size(0)
        zero_tail = torch.zeros((max_num_obj - num_obj, label.size(1)), dtype=label.dtype, device=label.device)
        label_with_tail = torch.cat((label, zero_tail), dim=0)
        labels_with_tail.append(label_with_tail)
    image_tensor = torch.stack(images)
    label_tensor = torch.stack(labels_with_tail)
    length_tensor = torch.tensor(lengths)

    if return_more:
        return image_tensor, label_tensor, length_tensor, scales, paddings
    return image_tensor, label_tensor, length_tensor
