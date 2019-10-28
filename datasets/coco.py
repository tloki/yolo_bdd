import numpy as np
import torch
from torchvision.datasets import CocoDetection

# from config_coco import NUM_ATTRIB, NUM_CLASSES_COCO, MISSING_IDS

from config_bdd100k import NUM_ATTRIB, NUM_CLASSES_COCO
#, MISSING_IDS

from .transforms import default_transform_fn, random_transform_fn
from utils import xywh_to_cxcywh
from datasets.image import _get_padding


class CocoDetectionBoundingBox(CocoDetection):

    def __init__(self, img_root, ann_file_name, img_size, transform='default', category='all', need_padding=False):


        super(CocoDetectionBoundingBox, self).__init__(img_root, ann_file_name)

        self._img_size = img_size

        self.need_padding = need_padding

        if transform == 'default':
            self._tf = default_transform_fn(img_size)
        elif transform == 'random':
            self._tf = random_transform_fn(img_size)
        else:
            raise ValueError("input transform can only be 'default' or 'random'.")
        if category == 'all':
            self.all_categories = True
            self.category_id = -1
        elif isinstance(category, int):
            self.all_categories = False
            self.category_id = category

    def __getitem__(self, index):
        img, targets = super(CocoDetectionBoundingBox, self).__getitem__(index)
        labels = []

        # loop over a list of detected objects
        for target in targets:
            bbox = torch.tensor(target['bbox'], dtype=torch.float32) # in xywh format
            category_id = target['category_id']

            # this should work!
            # from utils import pil_imshow
            # pil_imshow()
            # end debug

            if (not self.all_categories) and (category_id != self.category_id):
                continue
            one_hot_label = _coco_category_to_one_hot(category_id, dtype='float32')
            conf = torch.tensor([1.])
            label = torch.cat((bbox, conf, one_hot_label))
            labels.append(label)
        if labels:
            label_tensor = torch.stack(labels)
        else:
            label_tensor = torch.zeros((0, NUM_ATTRIB))
        transformed_img_tensor, label_tensor = self._tf(img, label_tensor)
        label_tensor = xywh_to_cxcywh(label_tensor)

        if self.need_padding:
            # Extract image
            # img = Image.open(img_path).convert("RGB")
            w, h = img.size
            max_size = max(w, h)
            _padding = _get_padding(h, w)
            # Add padding
            # _transform = default_transform_fn(_padding, self._img_size)
            # transformed_img_tensor, _ = self._transform(img)

            scale = self._img_size / max_size

            # image, percentage of image size to full supported size, padding part
            return transformed_img_tensor, label_tensor, label_tensor.size(0), scale, np.array(_padding)

        # image, label, label size
        return transformed_img_tensor, label_tensor, label_tensor.size(0)


def _coco_category_to_one_hot(category_id, dtype="uint"):
    """ convert from a category_id to one-hot vector, considering there are missing IDs in coco dataset."""
    new_id = _delete_coco_empty_category(category_id)
    return _category_to_one_hot(new_id, NUM_CLASSES_COCO, dtype)


def _category_to_one_hot(category_id, num_classes, dtype="uint"):
    """ convert from a category_id to one-hot vector """
    # if category_id == 0:
        # raise ValueError("0 in category id")
    return torch.from_numpy(np.eye(num_classes, dtype=dtype)[category_id])


def _delete_coco_empty_category(old_id):
    """The COCO dataset has 91 categories but 11 of them are empty.
    This function will convert the 80 existing classes into range [0-79].
    Note the COCO original class index starts from 1.
    The converted index starts from 0.
    Args:
        old_id (int): The category ID from COCO dataset.
    Return:
        new_id (int): The new ID after empty categories are removed. """
    return old_id
    starting_idx = 1
    new_id = old_id - starting_idx

    #
    # for missing_id in MISSING_IDS:
    #     if old_id > missing_id:
    #         new_id -= 1
    #     elif old_id == missing_id:
    #         raise KeyError("illegal category ID in coco dataset! ID # is {}".format(old_id))
    #     else:
    #         break
    return new_id
