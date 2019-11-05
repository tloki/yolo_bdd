import time
# from config_coco import NUM_CLASSES, EPSILON
from config_bdd100k import NUM_CLASSES, EPSILON
import argparse
import datetime
from utils import *
from models.yolov3 import load_yolov3_model
from datasets.utils import load_dataset
from config_bdd100k import *
from models.yolov3 import yolo_loss_fn
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from map_calc import MAP_Calculator


def run_yolo_evaluation(config: argparse.Namespace):
    # region logging
    # current_datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # log_file_name_by_time = current_datetime_str + ".log"
    # if config.debug:
    #     log_level = logging.DEBUG
    # elif config.verbose:
    #     log_level = logging.INFO
    # else:
    #     log_level = logging.WARNING
    # config_logging(config.log_dir, log_file_name_by_time, level=log_level)
    # endregion

    # set the device for inference
    device = config_device(config.cpu_only)

    # load model
    model = load_yolov3_model(config.weight_path, device, checkpoint=config.from_ckpt, mode='eval')
    # model = load_yolov3_model("yolov3_original.pt", device, checkpoint=False, mode='eval')
    # model = load_yolov3_model("ckpt_epoch_60.pt", device, checkpoint=False, mode='train')

    # model = model.to(device)

    # config.label_path
    # load data
    dataloader = load_dataset(type_=config.dataset_type,
                              img_dir=config.img_dir,
                              label_file=config.label_path,
                              img_size=config.img_size,
                              batch_size=config.batch_size,
                              n_cpu=config.n_cpu,
                              shuffle=False,
                              augment=False,
                              need_padding=True)

    run_evaluation(model=model, dataloader=dataloader, device=device, start_iter=config.start,
                   end_iter=config.stop, print_every=False,
                   conf_thres=0.8, nms_thres=0.4, class_path="data/bdd100k.names")



    return


def run_evaluation(model: nn.Module, dataloader, device, class_path, img_size=416, start_iter=None,
                   end_iter=None, print_every=False, conf_thres=0.8, nms_thres=0.4):

    class_names = load_class_names_from_file(class_path)
    map = MAP_Calculator(0.5, classes=class_names, noprint=True)

    for batch_i, batch in tqdm(enumerate(dataloader), total=end_iter):
        if batch_i < start_iter:
            continue

        if end_iter is not None and batch_i >= end_iter:
            break


        images = batch[0].to(device)
        scales = batch[3]
        paddings = batch[4]
        labels = batch[1].to(device)
        label_sizes = batch[2]


        images = images.to(device)
        labels = labels.to(device)
        # scales = scales.to(device)
        # paddings = paddings.to(device)

        model.train()
        with torch.no_grad():
            predictions = model(images)
        losses = yolo_loss_fn(predictions, labels, label_sizes, img_size, True)

        model.eval()
        with torch.no_grad():
            detections = model(images)

        l1, l2, l3, l4, l5 = losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item(), \
                             losses[4].item()

        # reference = detections
        # reference = labels

        labels = post_process(labels, True, conf_thres, nms_thres)
        detections = post_process(detections, True, conf_thres, nms_thres)

        # labels:
        # cx, cy, w, h, conf, class

        # show_results_as_images(detections, images, class_names)

        for label, detection in zip(labels, detections):
            cxcywh_to_xywh(detection)
            cxcywh_to_xywh(label)
            map.add_gt_pred_pair(label.to("cpu").numpy(), detection.to("cpu").numpy())

        results = list(zip(images, detections, scales, paddings))
        show_results_as_images(results, images, class_names)

        # region comments
        # reference = detections

        # for detection, scale, padding in zip(reference, scales, paddings):
            # detection[..., :4] = untransform_bboxes(detection[..., :4], 1, (0, 0))
            # cxcywh_to_xywh(detection)

        # just like save_img

        # img_path = '{}/{}/img'.format(config.out_dir, current_datetime_str)
        # make_output_dir(img_path)
        # results = list(zip(images, reference, scales, paddings))
        # show_results_as_images(results, images, class_names)
        # endregion

        if print_every:
            print("[Losses: total {}, coord {}, obj {}, noobj {}, class {}]".format(l1, l2, l3, l4, l5))

        l1 = losses[0].item()
        l2 = losses[1].item()
        l3 = losses[2].item()
        l4 = losses[3].item()
        l5 = losses[4].item()

        l1_sum += l1
        l2_sum += l2
        l3_sum += l3
        l4_sum += l4
        l5_sum += l5

    map_value, report = map.calculate()

    # print("l sums:", l1_sum, l2_sum, l3_sum, l4_sum, l5_sum)
    print("map, loss avgs:", map_value, l1_sum / n_iters, l2_sum / n_iters, l3_sum / n_iters, l4_sum / n_iters,
          l5_sum / n_iters)
    print(report)
    # print("end")


def show_results_as_images(results, image_tensor, class_names):
    # logging.info('Saving images:')
    # Iterate through images and save plot of detections
    for img_i, result in enumerate(results):
        path, detections, _, _ = result
        # logging.info("({}) Image: '{}'".format(img_i, path))
        img = image_tensor[img_i]
        # Create plot
        # img_output_filename = '{}/{}.png'.format(output_dir, img_i)
        show_det_image(detections, img, class_names)
    return


def show_det_image(detections, img_tensor, class_names):
    transform = ToPILImage()
    img = transform(img_tensor.to("cpu"))
    # Draw bounding boxes and labels of detections
    if detections is not None:
        img = draw_result(img, detections, class_names=class_names, show=True)
    return


# def post_process_gt(labels):
#     # (batch_size, )


def post_process(results_raw, nms, conf_thres, nms_thres):
    results = []
    for idx, result_raw in enumerate(results_raw):
        bboxes = result_raw[..., :4]
        scores = result_raw[..., 4]
        classes_one_hot = result_raw[..., 5:]
        classes = torch.argmax(classes_one_hot, dim=1)
        if nms:
            bboxes, scores, classes = \
                non_max_suppression(bboxes, scores, classes,
                                    num_classes=NUM_CLASSES,
                                    center=True,
                                    conf_thres=conf_thres,
                                    nms_thres=nms_thres)
        result = torch.cat((bboxes, scores.view((-1, 1)), classes.view((-1, 1)).float()), dim=1)
        results.append(result)
        logging.debug("The dimension of the result after nms is {} for idx {}".format(result.size(), idx))
    return results


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
            grouped_index[obj_class_].append(idx)
    return grouped_index


def iou(bbox1, bbox2, center=False):
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
    iou_ = area_intersect / (area1 + area2 - area_intersect + EPSILON)  # add epsilon to avoid NaN
    return iou_


def iou_one_to_many(bbox1, bboxes2, center=False):
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
    iou_ = area_intersect / (area1 + area2 - area_intersect + EPSILON)  # add epsilon to avoid NaN
    return iou_


def argsort(t, reverse=False):
    """Given a list, sort the list and return the original indices of the sorted list."""
    return sorted(range(len(t)), key=t.__getitem__, reverse=reverse)
