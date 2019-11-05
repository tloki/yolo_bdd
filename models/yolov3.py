import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# from config_coco import ANCHORS, NUM_ANCHORS_PER_SCALE, NUM_CLASSES, NUM_ATTRIB, LAST_LAYER_DIM
from config_bdd100k import ANCHORS, NUM_ANCHORS_PER_SCALE, NUM_CLASSES, NUM_ATTRIB, LAST_LAYER_DIM, EPSILON, \
    IGNORE_THRESH, NOOBJ_COEFF, COORD_COEFF

Tensor = torch.Tensor

def yolo_loss_fn(preds: Tensor, tgt: Tensor, tgt_len: Tensor, img_size: int, average=True):
    """Calculate the loss function given the predictions, the targets, the length of each target and the image size.
    Args:
        preds: (Tensor) the raw prediction tensor. Size is [B, N_PRED, NUM_ATTRIB],
                where B is the batch size;
                N_PRED is the total number of predictions, equivalent to 3*N_GRID*N_GRID for each scale;
                NUM_ATTRIB is the number of attributes, determined in config.py.
                coordinates is in format cx cy wh and is local (raw).
                objectness is in logit.
                class score is in logit.
        tgt:   (Tensor) the tensor of ground truths (targets). Size is [B, N_tgt_max, NUM_ATTRIB].
                where N_tgt_max is the max number of targets in this batch.
                If a certain sample has targets fewer than N_tgt_max, zeros are filled at the tail.
        tgt_len: (Tensor) a 1D tensor showing the number of the targets for each sample. Size is [B, ].
        img_size: (int) the size of the training image.
        average: (bool) the flag of whether the loss is summed loss or average loss over the batch size.
    Return:
        the total loss
        """

    # if average:
    #     red = "mean"
    # else:
    #     red = "sum"
    red = "sum"

    # generate the no-objectness mask. mask_noobj has size of [B, N_PRED]
    mask_noobj = noobj_mask_fn(preds, tgt)
    tgt_t_1d, idx_pred_obj = pre_process_targets(tgt, tgt_len, img_size)
    mask_noobj = noobj_mask_filter(mask_noobj, idx_pred_obj)

    # calculate the no-objectness loss
    pred_conf_logit = preds[..., 4]
    tgt_zero = torch.zeros(pred_conf_logit.size(), device=pred_conf_logit.device)
    # tgt_noobj = tgt_zero + (1 - mask_noobj) * 0.5
    pred_conf_logit = pred_conf_logit - (1 - mask_noobj) * 1e7
    noobj_loss = F.binary_cross_entropy_with_logits(pred_conf_logit, tgt_zero, reduction=red)

    # select the predictions corresponding to the targets
    n_batch, n_pred, _ = preds.size()
    preds_1d = preds.view(n_batch * n_pred, -1)
    preds_obj = preds_1d.index_select(0, idx_pred_obj)

    # calculate the coordinate loss
    coord_loss = F.mse_loss(preds_obj[..., :4], tgt_t_1d[..., :4], reduction=red)
    # assert not torch.isnan(coord_loss)

    # calculate the objectness loss
    pred_conf_obj_logit = preds_obj[..., 4]
    tgt_one = torch.ones(pred_conf_obj_logit.size(), device=pred_conf_obj_logit.device)
    obj_loss = F.binary_cross_entropy_with_logits(pred_conf_obj_logit, tgt_one, reduction=red)

    # calculate the classification loss
    class_loss = F.binary_cross_entropy_with_logits(preds_obj[..., 5:], tgt_t_1d[..., 5:], reduction=red)

    # total loss
    total_loss = noobj_loss * NOOBJ_COEFF + obj_loss + class_loss + coord_loss * COORD_COEFF

    if average:
        total_loss = total_loss / n_batch

    return total_loss, coord_loss, obj_loss, noobj_loss, class_loss


def noobj_mask_fn(pred: Tensor, target: Tensor):
    """pred is a 3D tensor with shape
    (num_batch, NUM_ANCHORS_PER_SCALE*num_grid*num_grid, NUM_ATTRIB). The raw data has been converted.
    target is a 3D tensor with shape
    (num_batch, max_num_object, NUM_ATTRIB).
     The max_num_objects depend on the sample which has max num_objects in this minibatch"""
    num_batch, num_pred, num_attrib = pred.size()
    assert num_batch == target.size(0)
    ious = iou_batch(pred[..., :4], target[..., :4], center=True) #in cxcywh format
    # for each pred bbox, find the target box which overlaps with it (without zero centered) most, and the iou value.
    max_ious, max_ious_idx = torch.max(ious, dim=2)
    noobj_indicator = torch.where((max_ious - IGNORE_THRESH) > 0, torch.zeros_like(max_ious), torch.ones_like(max_ious))
    return noobj_indicator


def noobj_mask_filter(mask_noobj: Tensor, idx_obj_1d: Tensor):
    n_batch, n_pred = mask_noobj.size()
    mask_noobj = mask_noobj.view(-1)
    filter_ = torch.zeros(mask_noobj.size(), device=mask_noobj.device)
    mask_noobj.scatter_(0, idx_obj_1d, filter_)
    mask_noobj = mask_noobj.view(n_batch, -1)
    return mask_noobj


def pre_process_targets(tgt: Tensor, tgt_len, img_size):
    """get the index of the predictions corresponding to the targets;
    and put targets from different sample into one dimension (flatten), getting rid of the tails;
    and convert coordinates to local.
    Args:
        tgt: (tensor) the tensor of ground truths (targets). Size is [B, N_tgt_max, NUM_ATTRIB].
                    where B is the batch size;
                    N_tgt_max is the max number of targets in this batch;
                    NUM_ATTRIB is the number of attributes, determined in config.py.
                    coordinates is in format cxcywh and is global.
                    If a certain sample has targets fewer than N_tgt_max, zeros are filled at the tail.
        tgt_len: (Tensor) a 1D tensor showing the number of the targets for each sample. Size is [B, ].
        img_size: (int) the size of the training image.
    :return
        tgt_t_flat: (tensor) the flattened and local target. Size is [N_tgt_total, NUM_ATTRIB],
                            where N_tgt_total is the total number of targets in this batch.
        idx_obj_1d: (tensor) the tensor of the indices of the predictions corresponding to the targets.
                            The size is [N_tgt_total, ]. Note the indices have been added the batch number,
                            therefore when the predictions are flattened, the indices can directly find the prediction.
    """
    # find the anchor box which has max IOU (zero centered) with the targets
    wh_anchor = torch.tensor(ANCHORS).to(tgt.device).float()
    n_anchor = wh_anchor.size(0)
    xy_anchor = torch.zeros((n_anchor, 2), device=tgt.device)
    bbox_anchor = torch.cat((xy_anchor, wh_anchor), dim=1)
    bbox_anchor.unsqueeze_(0)
    iou_anchor_tgt = iou_batch(bbox_anchor, tgt[..., :4], zero_center=True)
    _, idx_anchor = torch.max(iou_anchor_tgt, dim=1)

    # find the corresponding prediction's index for the anchor box with the max IOU
    strides_selection = [8, 16, 32]
    scale = idx_anchor // 3
    idx_anchor_by_scale = idx_anchor - scale * 3
    stride = 8 * 2 ** scale
    grid_x = (tgt[..., 0] // stride.float()).long()
    grid_y = (tgt[..., 1] // stride.float()).long()
    n_grid = img_size // stride
    large_scale_mask = (scale <= 1).long()
    med_scale_mask = (scale <= 0).long()
    idx_obj = \
        large_scale_mask * (img_size // strides_selection[2]) ** 2 * 3 + \
        med_scale_mask * (img_size // strides_selection[1]) ** 2 * 3 + \
        n_grid ** 2 * idx_anchor_by_scale + n_grid * grid_y + grid_x

    # calculate t_x and t_y
    t_x = (tgt[..., 0] / stride.float() - grid_x.float()).clamp(EPSILON, 1 - EPSILON)
    t_x = torch.log(t_x / (1. - t_x))   #inverse of sigmoid
    t_y = (tgt[..., 1] / stride.float() - grid_y.float()).clamp(EPSILON, 1 - EPSILON)
    t_y = torch.log(t_y / (1. - t_y))    # inverse of sigmoid

    # calculate t_w and t_h
    w_anchor = wh_anchor[..., 0]
    h_anchor = wh_anchor[..., 1]
    w_anchor = torch.index_select(w_anchor, 0, idx_anchor.view(-1)).view(idx_anchor.size())
    h_anchor = torch.index_select(h_anchor, 0, idx_anchor.view(-1)).view(idx_anchor.size())
    t_w = torch.log((tgt[..., 2] / w_anchor).clamp(min=EPSILON))
    t_h = torch.log((tgt[..., 3] / h_anchor).clamp(min=EPSILON))

    # the raw target tensor
    tgt_t = tgt.clone().detach()

    tgt_t[..., 0] = t_x
    tgt_t[..., 1] = t_y
    tgt_t[..., 2] = t_w
    tgt_t[..., 3] = t_h

    # aggregate processed targets and the corresponding prediction index from different batches in to one dimension
    n_batch = tgt.size(0)
    n_pred = sum([(img_size // s) ** 2 for s in strides_selection]) * 3

    idx_obj_1d = []
    tgt_t_flat = []

    for i_batch in range(n_batch):
        v = idx_obj[i_batch]
        t = tgt_t[i_batch]
        l = tgt_len[i_batch]
        idx_obj_1d.append(v[:l] + i_batch * n_pred)
        tgt_t_flat.append(t[:l])

    idx_obj_1d = torch.cat(idx_obj_1d)
    tgt_t_flat = torch.cat(tgt_t_flat)

    return tgt_t_flat, idx_obj_1d


def iou_batch(bboxes1: Tensor, bboxes2: Tensor, center=False, zero_center=False):
    """Calculate the IOUs between bboxes1 and bboxes2.
    :param
      bboxes1: (Tensor) A 3D tensor representing first group of bboxes.
        The dimension is (B, N1, 4). B is the number of samples in the batch.
        N1 is the number of bboxes in each sample.
        The third dimension represent the bbox, with coordinate (x, y, w, h) or (cx, cy, w, h).
      bboxes2: (Tensor) A 3D tensor representing second group of bboxes.
        The dimension is (B, N2, 4). It is similar to bboxes1, the only difference is N2.
        N1 is the number of bboxes in each sample.
      center: (bool). Whether the bboxes are in format (cx, cy, w, h).
      zero_center: (bool). Whether to align two bboxes so their center is aligned.
    :return
      iou_: (Tensor) A 3D tensor representing the IOUs.
        The dimension is (B, N1, N2)."""
    x1 = bboxes1[..., 0]
    y1 = bboxes1[..., 1]
    w1 = bboxes1[..., 2]
    h1 = bboxes1[..., 3]

    x2 = bboxes2[..., 0]
    y2 = bboxes2[..., 1]
    w2 = bboxes2[..., 2]
    h2 = bboxes2[..., 3]

    area1 = w1 * h1
    area2 = w2 * h2

    if zero_center:
        w1.unsqueeze_(2)
        w2.unsqueeze_(1)
        h1.unsqueeze_(2)
        h2.unsqueeze_(1)
        w_intersect = torch.min(w1, w2).clamp(min=0)
        h_intersect = torch.min(h1, h2).clamp(min=0)
    else:
        if center:
            x1 = x1 - w1 / 2
            y1 = y1 - h1 / 2
            x2 = x2 - w2 / 2
            y2 = y2 - h2 / 2
        right1 = (x1 + w1).unsqueeze(2)
        right2 = (x2 + w2).unsqueeze(1)
        top1 = (y1 + h1).unsqueeze(2)
        top2 = (y2 + h2).unsqueeze(1)
        left1 = x1.unsqueeze(2)
        left2 = x2.unsqueeze(1)
        bottom1 = y1.unsqueeze(2)
        bottom2 = y2.unsqueeze(1)
        w_intersect = (torch.min(right1, right2) - torch.max(left1, left2)).clamp(min=0)
        h_intersect = (torch.min(top1, top2) - torch.max(bottom1, bottom2)).clamp(min=0)
    area_intersect = h_intersect * w_intersect

    iou_ = area_intersect / (area1.unsqueeze(2) + area2.unsqueeze(1) - area_intersect + EPSILON)

    return iou_


class ConvLayer(nn.Module):
    """Basic 'conv' layer, including:
     A Conv2D layer with desired channels and kernel size,
     A batch-norm layer,
     and A leakyReLu layer with neg_slope of 0.1.
     (Didn't find too much resource what neg_slope really is.
     By looking at the darknet source code, it is confirmed the neg_slope=0.1.
     Ref: https://github.com/pjreddie/darknet/blob/master/src/activations.h)
     Please note here we distinguish between Conv2D layer and Conv layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, lrelu_neg_slope=0.1):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_neg_slope)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)

        return out


class ResBlock(nn.Module):
    """The basic residual block used in YoloV3.
    Each ResBlock consists of two ConvLayers and the input is added to the final output.
    In YoloV3 paper, the first convLayer has half of the number of the filters as much as the second convLayer.
    The first convLayer has filter size of 1x1 and the second one has the filter size of 3x3.
    """

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        assert in_channels % 2 == 0  # ensure the in_channels is an even number.
        half_in_channels = in_channels // 2
        self.conv1 = ConvLayer(in_channels, half_in_channels, 1)
        self.conv2 = ConvLayer(half_in_channels, in_channels, 3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual

        return out


def make_conv_and_res_block(in_channels, out_channels, res_repeat):
    """In Darknet 53 backbone, there is usually one Conv Layer followed by some ResBlock.
    This function will make that.
    The Conv layers always have 3x3 filters with stride=2.
    The number of the filters in Conv layer is the same as the out channels of the ResBlock"""
    model = nn.Sequential()
    model.add_module('conv', ConvLayer(in_channels, out_channels, 3, stride=2))
    for idx in range(res_repeat):
        model.add_module('res{}'.format(idx), ResBlock(out_channels))
    return model


class YoloLayer(nn.Module):

    def __init__(self, scale, stride):
        super(YoloLayer, self).__init__()
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            idx = None
        self.anchors = torch.tensor([ANCHORS[i] for i in idx])
        self.stride = stride

    def forward(self, x):
        num_batch = x.size(0)
        num_grid = x.size(2)

        if self.training:
            output_raw = x.view(num_batch,
                                NUM_ANCHORS_PER_SCALE,
                                NUM_ATTRIB,
                                num_grid,
                                num_grid).permute(0, 1, 3, 4, 2).contiguous().view(num_batch, -1, NUM_ATTRIB)
            return output_raw
        else:
            prediction_raw = x.view(num_batch,
                                    NUM_ANCHORS_PER_SCALE,
                                    NUM_ATTRIB,
                                    num_grid,
                                    num_grid).permute(0, 1, 3, 4, 2).contiguous()

            self.anchors = self.anchors.to(x.device).float()
            # Calculate offsets for each grid
            grid_tensor = torch.arange(num_grid, dtype=torch.float, device=x.device).repeat(num_grid, 1)
            grid_x = grid_tensor.view([1, 1, num_grid, num_grid])
            grid_y = grid_tensor.t().view([1, 1, num_grid, num_grid])
            anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
            anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))

            # Get outputs
            x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * self.stride  # Center x
            y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * self.stride  # Center y
            w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w  # Width
            h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h  # Height
            bbox_pred = torch.stack((x_center_pred, y_center_pred, w_pred, h_pred), dim=4).view(
                (num_batch, -1, 4))  # cxcywh
            conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)  # Conf
            cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, NUM_CLASSES)  # Cls pred one-hot.

            output = torch.cat((bbox_pred, conf_pred, cls_pred), -1)
            return output


class DetectionBlock(nn.Module):
    """The DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
    1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n,
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)
    out_channels = n
    """

    def __init__(self, in_channels, out_channels, scale, stride):
        super(DetectionBlock, self).__init__()
        assert out_channels % 2 == 0  # assert out_channels is an even number
        half_out_channels = out_channels // 2
        self.conv1 = ConvLayer(in_channels, half_out_channels, 1)
        self.conv2 = ConvLayer(half_out_channels, out_channels, 3)
        self.conv3 = ConvLayer(out_channels, half_out_channels, 1)
        self.conv4 = ConvLayer(half_out_channels, out_channels, 3)
        self.conv5 = ConvLayer(out_channels, half_out_channels, 1)
        self.conv6 = ConvLayer(half_out_channels, out_channels, 3)
        self.conv7 = nn.Conv2d(out_channels, LAST_LAYER_DIM, 1, bias=True)
        self.yolo = YoloLayer(scale, stride)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        self.branch = self.conv5(tmp)
        tmp = self.conv6(self.branch)
        tmp = self.conv7(tmp)
        out = self.yolo(tmp)

        return out


class DarkNet53BackBone(nn.Module):

    def __init__(self):
        super(DarkNet53BackBone, self).__init__()
        self.conv1 = ConvLayer(3, 32, 3)
        self.cr_block1 = make_conv_and_res_block(32, 64, 1)
        self.cr_block2 = make_conv_and_res_block(64, 128, 2)
        self.cr_block3 = make_conv_and_res_block(128, 256, 8)
        self.cr_block4 = make_conv_and_res_block(256, 512, 8)
        self.cr_block5 = make_conv_and_res_block(512, 1024, 4)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.cr_block1(tmp)
        tmp = self.cr_block2(tmp)
        out3 = self.cr_block3(tmp)
        out2 = self.cr_block4(out3)
        out1 = self.cr_block5(out2)

        return out1, out2, out3


class YoloNetTail(nn.Module):
    """The tail side of the YoloNet.
    It will take the result from DarkNet53BackBone and do some upsampling and concatenation.
    It will finally output the detection result.
    Assembling YoloNetTail and DarkNet53BackBone will give you final result"""

    def __init__(self):
        super(YoloNetTail, self).__init__()
        self.detect1 = DetectionBlock(1024, 1024, 'l', 32)
        self.conv1 = ConvLayer(512, 256, 1)
        self.detect2 = DetectionBlock(768, 512, 'm', 16)
        self.conv2 = ConvLayer(256, 128, 1)
        self.detect3 = DetectionBlock(384, 256, 's', 8)

    def forward(self, x1, x2, x3):
        out1 = self.detect1(x1)
        branch1 = self.detect1.branch
        tmp = self.conv1(branch1)
        tmp = F.interpolate(tmp, scale_factor=2)
        tmp = torch.cat((tmp, x2), 1)
        out2 = self.detect2(tmp)
        branch2 = self.detect2.branch
        tmp = self.conv2(branch2)
        tmp = F.interpolate(tmp, scale_factor=2)
        tmp = torch.cat((tmp, x3), 1)
        out3 = self.detect3(tmp)

        return out1, out2, out3


class YoloNetV3(nn.Module):

    def __init__(self, nms=False, post=True):
        super(YoloNetV3, self).__init__()
        self.darknet = DarkNet53BackBone()
        self.yolo_tail = YoloNetTail()
        self.nms = nms
        self._post_process = post

    def forward(self, x):
        tmp1, tmp2, tmp3 = self.darknet(x)
        out1, out2, out3 = self.yolo_tail(tmp1, tmp2, tmp3)
        out = torch.cat((out1, out2, out3), 1)
        # logging.debug("The dimension of the output before nms is {}".format(out.size())) lol
        return out

    def yolo_last_layers(self):
        _layers = [self.yolo_tail.detect1.conv7,
                   self.yolo_tail.detect2.conv7,
                   self.yolo_tail.detect3.conv7]
        return _layers

    def yolo_last_two_layers(self):
        _layers = self.yolo_last_layers() + \
                  [self.yolo_tail.detect1.conv6,
                   self.yolo_tail.detect2.conv6,
                   self.yolo_tail.detect3.conv6]
        return _layers

    def yolo_last_three_layers(self):
        _layers = self.yolo_last_two_layers() + \
                  [self.yolo_tail.detect1.conv5,
                   self.yolo_tail.detect2.conv5,
                   self.yolo_tail.detect3.conv5]
        return _layers

    def yolo_tail_layers(self):
        _layers = [self.yolo_tail]
        return _layers

    def yolo_last_n_layers(self, n):
        try:
            n = int(n)
        except ValueError:
            pass
        if n == 1:
            return self.yolo_last_layers()
        elif n == 2:
            return self.yolo_last_two_layers()
        elif n == 3:
            return self.yolo_last_three_layers()
        elif n == 'tail':
            return self.yolo_tail_layers()
        else:
            raise ValueError("n>3 not defined")


def load_yolov3_model(weight_path, device, checkpoint=False, mode='eval', only_darknet=False) -> nn.Module:
    yolo_model = YoloNetV3(nms=True)
    weights = torch.load(weight_path, map_location=device)

    if only_darknet:
        from copy import deepcopy
        new_weights = deepcopy(weights)
        for key in weights:
            if not key.startswith("darknet"):
                del new_weights[key]

        weights = new_weights

    # TODO: this throws a KeyError: 'model_state_dict'
    if not checkpoint:
        yolo_model.load_state_dict(weights, strict=False)
    else:
        yolo_model.load_state_dict(weights['model_state_dict'], strict=False)
    yolo_model.to(device)

    # TODO: fix ckpt-dir argument
    # if checkpoint:
    #     logging.info("loading checkpoint from file '{}'".format(weight_path))
    #     yolo_model.load_state_dict(torch.load(weight_path)['model_state_dict'], strict=False)
    #     _model.load_state_dict(torch.load(weight_path))

    if mode == 'eval':
        yolo_model.eval()
    elif mode == 'train':
        yolo_model.train()
    else:
        raise ValueError("YoloV3 model can be only loaded in 'train' or 'eval' mode.")
    return yolo_model

def post_process_custom(results_raw, nms, conf_thres, nms_thres, num_classes, only_classes=None, group=False, group_rest=None):
    results = []

    # if len(results_raw) == 0:
    #     return []

    for idx, result_raw in enumerate(results_raw):
        bboxes = result_raw[..., :4]
        scores = result_raw[..., 4]
        classes_one_hot = result_raw[..., 5:]

        classes = torch.argmax(classes_one_hot, dim=1)
        if nms:
            bboxes, scores, classes = \
                non_max_suppression(bboxes, scores, classes,
                                    num_classes=num_classes,
                                    center=True,
                                    conf_thres=conf_thres,
                                    nms_thres=nms_thres)

        scores_flatten = scores.view((-1, 1))
        classes_flatten = classes.view((-1, 1)).float()
        result = torch.cat((bboxes, scores_flatten, classes_flatten), dim=1)

        results.append(result)

    # 0 - object of interest
    if only_classes is not None and only_classes is not False:
        if group_rest is not None and group_rest is not False:
            assert group is True
            results_yes = filter_classes_of_interest(results, only_classes)


            for y in results_yes:
                if len(y) == 0:
                    continue
                y[..., 5:] = 0

            results_no = filter_classes_of_interest(results, only_classes, inverse=True)

            for n in results_no:
                if len(n) == 0:
                    continue
                n[..., 5:] = 1

            results = []
            # for y, n in zip(results_yes, results_no):
            #     results.append(torch.cat((y, n), 0))

            results.extend(results_yes)
            results.extend(results_no)

            return results

        if group is None or not group:
            return filter_classes_of_interest(results, only_classes)
        else:
            results = filter_classes_of_interest(results, only_classes)
            for r in results:
                if len(r) == 0:
                    continue
                r[..., 5:] = 0
            return results

    return results

def filter_classes_of_interest(detections, class_list, tol=0.25, inverse=False):

    if class_list is None or class_list is False:
        return detections

    filtered_detections = []

    for detection in detections:
        if len(detection) == 0:
            filtered_detections.append(detection)
            continue

        classes_one_hot = detection[..., 5:]

        requirement = torch.zeros_like(classes_one_hot, dtype=torch.bool)

        if not inverse:
            for class_index in class_list:
                requirement += (classes_one_hot >= class_index - 1 - tol) * (classes_one_hot <= class_index - 1 + tol)
        else:
            for class_index in class_list:
                requirement += (classes_one_hot <= class_index - 1 - tol) + (classes_one_hot >= class_index - 1 + tol)

        requirement = requirement.reshape(-1)

        detection = detection[requirement]

        if len(detection) > 0:
            filtered_detections.append(detection)
        else:
            filtered_detections.append([])

    return filtered_detections

def non_max_suppression(bboxes, scores, classes, num_classes, conf_thres=0.8, nms_thres=0.5, center=False):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        bboxes: (tensor) The location predictions for the img, Shape: [num_priors,4].
        scores: (tensor) The class prediction scores for the img, Shape:[num_priors].
        classes: (tensor) The label (non-one-hot) representation of the classes of the objects,
          Shape: [num_priors].
        num_classes: (int) The number of all the classes.
        conf_thres: (float) Threshold where all the detections below this value will be ignored.
        nms_thres: (float) The overlap thresh for suppressing unnecessary boxes.
        center: (boolean) Whether the bboxes format is cxcywh or xywh.
    Return:
        The indices of the kept boxes with respect to num_priors, and they are always in xywh format.
    """

    # make sure bboxes and scores have the same 0th dimension
    assert bboxes.shape[0] == scores.shape[0] == classes.shape[0]
    num_prior = bboxes.shape[0]

    # if no objects, return raw result
    if num_prior == 0:
        return bboxes, scores, classes

    # threshold out low confidence detection

    if conf_thres > 0:
        conf_index = torch.nonzero(torch.ge(scores, conf_thres)).squeeze()

        bboxes = bboxes.index_select(0, conf_index)
        scores = scores.index_select(0, conf_index)
        classes = classes.index_select(0, conf_index)

    # print("nms: num classes", num_classes)
    # if there are multiple classes, divide them into groups
    grouped_indices = group_same_class_object(classes, one_hot=False, num_classes=num_classes)
    selected_indices_final = []

    for class_id, member_idx in enumerate(grouped_indices):
        member_idx_tensor = bboxes.new_tensor(member_idx, dtype=torch.long)
        bboxes_one_class = bboxes.index_select(dim=0, index=member_idx_tensor)
        scores_one_class = scores.index_select(dim=0, index=member_idx_tensor)
        scores_one_class, sorted_indices = torch.sort(scores_one_class, descending=False)

        selected_indices = []

        while sorted_indices.size(0) != 0:
            picked_index = sorted_indices[-1]
            selected_indices.append(picked_index)
            picked_bbox = bboxes_one_class[picked_index]
            ious = iou_one_to_many(picked_bbox, bboxes_one_class[sorted_indices[:-1]], center=center)
            under_indices = torch.nonzero(ious <= nms_thres).squeeze()
            sorted_indices = sorted_indices.index_select(dim=0, index=under_indices)

        selected_indices_final.extend([member_idx[i] for i in selected_indices])

    selected_indices_final = bboxes.new_tensor(selected_indices_final, dtype=torch.long)
    bboxes_result = bboxes.index_select(dim=0, index=selected_indices_final)
    scores_result = scores.index_select(dim=0, index=selected_indices_final)
    classes_result = classes.index_select(dim=0, index=selected_indices_final)

    return bboxes_result, scores_result, classes_result

def group_same_class_object(obj_classes, one_hot=True, num_classes=-1):
    """
    Given a list of class results, group the object with the same class into a list.
    Returns a list with the length of num_classes, where each bucket has the objects with the same class.
    :param
        obj_classes: (torch.tensor) The representation of classes of object.
         It can be either one-hot or label (non-one-hot).
         If it is one-hot, the shape should be: [num_objects, num_classes]
         If it is label (non-non-hot), the shape should be: [num_objects, ]
        one_hot: (bool) A flag telling the function whether obj_classes is one-hot representation.
        num_classes: (int) The max number of classes if obj_classes is represented as non-one-hot format.
    :return:
        a list of of a list, where for the i-th list,
        the elements in such list represent the indices of the objects in class i.
    """
    # print("obj classes:", obj_classes)
    # print("num classes:", num_classes)
    if one_hot:
        num_classes = obj_classes.shape[-1]
    else:
        assert num_classes != -1
    grouped_index = [[] for _ in range(num_classes)]
    if one_hot:
        for idx, class_one_hot in enumerate(obj_classes):
            grouped_index[torch.argmax(class_one_hot)].append(idx)
    else:
        for idx, obj_class_ in enumerate(obj_classes):
            # print(obj_class_)
            grouped_index[obj_class_].append(idx)
    return grouped_index


def iou(bbox1, bbox2, center=False, epsilon=1e-9):
    """Calculate IOU for two bboxes. If center is false, then they should all in xywh format.
    Else, they should all be in cxcywh format"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    if center:
        x1 = x1 - w1 / 2
        y1 = y1 - h1 / 2
        x2 = x2 - w2 / 2
        y2 = y2 - h2 / 2
    area1 = w1 * h1
    area2 = w2 * h2
    right1 = x1 + w1
    right2 = x2 + w2
    bottom1 = y1 + h1
    bottom2 = y2 + h2
    w_intersect = (torch.min(right1, right2) - torch.max(x1, x2)).clamp(min=0)
    h_intersect = (torch.min(bottom1, bottom2) - torch.max(y1, y2)).clamp(min=0)
    area_intersect = w_intersect * h_intersect
    iou_ = area_intersect / (area1 + area2 - area_intersect + epsilon)  # add epsilon to avoid NaN
    return iou_


def iou_one_to_many(bbox1, bboxes2, center=False, epsilon=1e-9):
    """Calculate IOU for one bbox with another group of bboxes.
    If center is false, then they should all in xywh format.
    Else, they should all be in cxcywh format"""
    x1, y1, w1, h1 = bbox1
    x2 = bboxes2[..., 0]
    y2 = bboxes2[..., 1]
    w2 = bboxes2[..., 2]
    h2 = bboxes2[..., 3]
    if center:
        x1 = x1 - w1 / 2
        y1 = y1 - h1 / 2
        x2 = x2 - w2 / 2
        y2 = y2 - h2 / 2
    area1 = w1 * h1
    area2 = w2 * h2
    right1 = x1 + w1
    right2 = x2 + w2
    bottom1 = y1 + h1
    bottom2 = y2 + h2
    w_intersect = (torch.min(right1, right2) - torch.max(x1, x2)).clamp(min=0)
    h_intersect = (torch.min(bottom1, bottom2) - torch.max(y1, y2)).clamp(min=0)
    area_intersect = w_intersect * h_intersect
    iou_ = area_intersect / (area1 + area2 - area_intersect + epsilon)  # add epsilon to avoid NaN
    return iou_

def cxcywh_to_xywh(bbox):
    if isinstance(bbox, list):
        for b in bbox:
            cxcywh_to_xywh(b)
        return

    if len(bbox) == 0:
        return bbox

    bbox[..., 0] -= bbox[..., 2] / 2
    bbox[..., 1] -= bbox[..., 3] / 2
    return bbox


def upscale_detections(detections, input_width, input_height, output_width, output_height, no_padding=False,
                       relative=True):
    # assert input_width == input_height

    if len(detections) == 0 or detections is None:
        return detections

    has_no_detections = True
    for d in detections:
        if len(d) > 0:
            has_no_detections = False
            break

    if has_no_detections:
        return detections

    upscaled_detections = []
    for det in detections:
        if len(det) == 0:
            upscaled_detections.append([])
            continue

        x1 = det[:, 0:1]
        y1 = det[:, 1:2]
        x2 = det[:, 2:3]
        y2 = det[:, 3:4]
        rest = det[:, 4:]

        x_upscale = output_width / input_width
        y_upscale = output_height / input_height

        # absolute
        if not relative:
            raise NotImplementedError("not implemented yet")

        # relative
        else:
            # print("relative")
            w, h = x2, y2
            if output_width == output_height or no_padding:
                # no padding
                x1 = x1 * x_upscale
                y1 = y1 * y_upscale
                w = w * x_upscale
                h = h * y_upscale

            elif output_width > output_height:
                # print("widescreen")
                # vertical padding took place
                x1 = x1 * x_upscale
                y1 = y1 * x_upscale
                w = x2 * x_upscale
                h = y2 * x_upscale

                padded_output_height = x_upscale * input_height
                y_offset = (padded_output_height - output_height) / 2

                y1 -= y_offset

            else:
                x1 = x1 * y_upscale
                y1 = y1 * y_upscale
                w = x2 * y_upscale
                h = y2 * y_upscale

                padded_output_width = x_upscale * input_width
                x_offset = (padded_output_width - output_width) / 2

                x1 -= x_offset
        upscaled_detections.append(torch.cat((x1, y1, w, h), -1))
    return upscaled_detections

import cv2

def imshow_with_detections(img, detections, is_rgb=True, key_value=1, relative=True, show=True, return_frame=False):
    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if detections is None or len(detections) == 0:
        if show:
            cv2.imshow("frame", img)
            key = cv2.waitKey(key_value)
            return key

    # for dets in detections:
    #     print("ok")
    #     if len(dets) == 0:
    #         continue

    x1 = detections[:, 0:1]
    y1 = detections[:, 1:2]
    w = detections[:, 2:3]
    h = detections[:, 3:4]

    if relative:
        x2 = x1 + w
        y2 = y1 + h
    else:
        x2 = w
        y2 = h

    len_dets = detections.size()[0]
    for det in range(len_dets):
        x1_ = round(x1[det, 0].item())
        y1_ = round(y1[det, 0].item())
        x2_ = round(x2[det, 0].item())
        y2_ = round(y2[det, 0].item())

        cv2.rectangle(img, (x1_, y1_), (x2_, y2_), (0, 0, 255), 2)

    if show:
        cv2.imshow("frame", img)
        key = cv2.waitKey(key_value)
        if not return_frame:
            return key
    else:
        if return_frame:
            return  img
        else:
            return None

    return None