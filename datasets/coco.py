import numpy as np
import torch
from torchvision.datasets import CocoDetection


from .transforms import default_transform_fn, random_transform_fn
from utils.utils import xywh_to_cxcywh


class CocoDetectionBoundingBox(CocoDetection):

    def __init__(self, img_root, ann_file_name, img_size, num_classes=10, transform='default', category='all',
                 missing_classes=None, group_other=False, category_dict=None, group_selected=False):
        super(CocoDetectionBoundingBox, self).__init__(img_root, ann_file_name)
        self._img_size = img_size
        if transform == 'default':
            self._tf = default_transform_fn(img_size)
        elif transform == 'random':
            self._tf = random_transform_fn(img_size)
        else:
            raise ValueError("input transform can only be 'default' or 'random'.")

        self.group_other = group_other
        # if group_other:
        #     raise NotImplementedError("grouping not yet implemented")

        self.num_classes = num_classes
        self.group_selected = group_selected

        if category == 'all':
            self.all_categories = True
            self.category_ids = -1
        elif isinstance(category, int):
            self.all_categories = False
            self.category_ids = [category]
        elif isinstance(category, str):
            self.category_ids = [category_dict[category]]
        elif isinstance(category, list):
            if isinstance(category[0], int):
                self.all_categories = False
                self.category_ids = category
            elif isinstance(category[0], str):
                self.all_categories = False
                self.category_ids = [category_dict[cat_name] for cat_name in category]
        else:
            raise ValueError("incorrect category argument")

        self.map_categories = None
        if self.all_categories == False:
            self.map_categories = {cat: i for i, cat in enumerate(self.category_ids)}

        # if self.category_ids != "all" and len(self.category_ids) > 0:
        #

        self.missing_classes = missing_classes

    def __getitem__(self, index):
        # print("index", index)
        img, targets = super(CocoDetectionBoundingBox, self).__getitem__(index)

        # print("target:", targets)

        #print("index", index)
        labels = []

        # loop over a list of detected objects
        for target_i, target in enumerate(targets):
            # print("target i:", target_i)
            bbox = torch.tensor(target['bbox'], dtype=torch.float32)  # in xywh format
            category_id = target['category_id']

            if not self.all_categories:
                category_id = _delete_coco_empty_category(category_id, self.missing_classes)
                if category_id in self.category_ids:

                    category_id = self.map_categories[category_id]
                    if self.group_selected:
                        one_hot_label = _category_to_one_hot(category_id=1,
                                                             num_classes=1 + int(self.group_other),
                                                             dtype='float32')
                    else:
                        one_hot_label = _category_to_one_hot(category_id=category_id + int(self.group_other),
                                                             num_classes=len(self.category_ids) + int(self.group_other),
                                                             dtype='float32')
                else:
                    if not self.group_other:
                        continue
                    if self.group_selected:
                        one_hot_label = _category_to_one_hot(category_id=0,
                                                             num_classes=2,
                                                             dtype='float32')
                    else:
                        one_hot_label = _category_to_one_hot(category_id=0,
                                                             num_classes=1 + len(self.category_ids),
                                                             dtype='float32')


            else:
                one_hot_label = _coco_category_to_one_hot(category_id=category_id,
                                                          num_classes=self.num_classes,
                                                          dtype='float32',
                                                          missing_classes=self.missing_classes)
            conf = torch.tensor([1.])
            label = torch.cat((bbox, conf, one_hot_label))
            labels.append(label)

        label_tensor_empty = None
        label_tensor = None
        result_empty = False
        if labels:
            label_tensor = torch.stack(labels)
        else:
            result_empty = True
            if self.all_categories:
                label_tensor_empty = torch.zeros((1, self.num_classes + 5))  # was torch.zeros((0, self.num_classes))
            else:
                if self.group_selected:
                    label_tensor_empty = torch.zeros((1, 1 + int(self.group_other) + 5))
                else:
                    label_tensor_empty = torch.zeros((1, len(self.category_ids) + int(self.group_other) + 5))

        if result_empty:
            transformed_img_tensor, _ = self._tf(img, label_tensor_empty)
            return transformed_img_tensor, label_tensor_empty, label_tensor_empty.size(0)

        transformed_img_tensor, label_tensor = self._tf(img, label_tensor)
        label_tensor = xywh_to_cxcywh(label_tensor)

        return transformed_img_tensor, label_tensor, label_tensor.size(0)


def _coco_category_to_one_hot(category_id, num_classes: int, dtype="uint", missing_classes=None):
    """ convert from a category_id to one-hot vector, considering there are missing IDs in coco dataset."""
    new_id = _delete_coco_empty_category(category_id, missing_classes)
    return _category_to_one_hot(new_id, num_classes, dtype)


def _category_to_one_hot(category_id, num_classes, dtype="uint"):
    """ convert from a category_id to one-hot vector """
    # if category_id == 0:
        # raise ValueError("0 in category id")
    return torch.from_numpy(np.eye(num_classes, dtype=dtype)[category_id - 1]) # loki: -1


def _delete_coco_empty_category(old_id, missing_ids=None):
    """The COCO dataset has 91 categories but 11 of them are empty.
    This function will convert the 80 existing classes into range [0-79].
    Note the COCO original class index starts from 1.
    The converted index starts from 0.
    [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
    Args:
        old_id (int): The category ID from COCO dataset.
    Return:
        new_id (int): The new ID after empty categories are removed. """
    if missing_ids is None:
        missing_ids = []

    # return old_id
    starting_idx = 0
    new_id = old_id - starting_idx

    for missing_id in missing_ids:
        if old_id > missing_id:
            new_id -= 1
        elif old_id == missing_id:
            raise KeyError("illegal category ID in coco dataset! ID # is {}".format(old_id))
        else:
            break
    return new_id


def get_num_scales(anchors):
    return len(anchors)
