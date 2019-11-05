import torch
import torch.nn as nn
import os
import logging

# from config_bdd100k import MISSING_IDS

import json
from PIL import ImageDraw, ImageFont, Image
from torchvision.transforms import ToPILImage


def get_gradients(model: nn.Module):
    # https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    layers = [module for module in model.modules() if type(module) != nn.Sequential]
    for i, layer in enumerate(layers):
        out = ""
        out = "layer {}.\t[{}]\t".format(i + 1, type(layer).__name__)

        if hasattr(layer, 'weight') and layer.weight is not None:
            out += "weights grads: {}\t".format(layer.weight.grad)
        else:
            out += "[no weights]\t"

        if hasattr(layer, 'bias') and layer.bias is not None:
            out += "bias grads: {}\t".format(layer.bias.grad)
        else:
            out += "[no bias]\t"
        out.expandtabs(10)
        print(out)

    return


def get_max_gradients_per_layer(model: nn.Module, print_all=False):
    # https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    layers = [module for module in model.modules() if type(module) != nn.Sequential]
    print()

    max_gradient_weight = 0
    max_gradient_bias = 0

    for i, layer in enumerate(layers):
        out = "layer {:3d}. [{}]\t".format(i + 1, type(layer).__name__)

        if hasattr(layer, 'weight') and layer.weight is not None:
            max_weight = torch.max(torch.abs(layer.weight.grad))
            max_gradient_weight = max(max_gradient_weight, max_weight)
            out += "weights grads: {:.3f}".format(max_weight)
        else:
            out += "weights grads: [no weights]"
        out += "\t"

        if hasattr(layer, 'bias') and layer.bias is not None:
            max_weight = torch.max(torch.abs(layer.bias.grad))
            max_gradient_bias = max(max_gradient_bias, max_weight)
            out += "bias grads: {:.3f}".format(max_weight)
        else:
            out += "bias grads: [no bias]"

        out += "\t"
        out = out.expandtabs(18)

        if print_all:
            print(out)

    return float(max_gradient_weight.cpu().numpy()), \
           float(max_gradient_bias.cpu().numpy())


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def save_results_as_json(results, json_path):
    results_json = []
    for result_raw in results:
        path, detections, _, _ = result_raw
        image_id = os.path.basename(path)
        image_id, _ = os.path.splitext(image_id)
        try:
            image_id = int(image_id)
        except ValueError:
            pass
        for detection in detections:
            detection = detection.tolist()
            bbox = detection[:4]
            score = detection[4]
            category_id = add_coco_empty_category(int(detection[5]))
            result = {'image_id': image_id, 'category_id': category_id, 'bbox': bbox, 'score': score}
            results_json.append(result)
    with open(json_path, 'w') as f:
        json.dump(results_json, f)
    return


def save_det_image(img_path, detections, output_img_path, class_names):
    img = Image.open(img_path)
    # Draw bounding boxes and labels of detections
    if detections is not None:
        img = draw_result(img, detections, class_names=class_names)
    img.save(output_img_path)
    return


def save_results_as_images(results, output_dir, class_names):
    logging.info('Saving images:')
    # Iterate through images and save plot of detections
    for img_i, result in enumerate(results):
        path, detections, _, _ = result
        logging.info("({}) Image: '{}'".format(img_i, path))
        # Create plot
        img_output_filename = '{}/{}.png'.format(output_dir, img_i)
        save_det_image(path, detections, img_output_filename, class_names)
    return


def save_checkpoint_weight_file(model, optimizer, epoch, batch, loss, weight_file_path):
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, weight_file_path)
    logging.info("saving model at epoch {}, batch {} to {}".format(epoch, batch, weight_file_path))
    return


def make_output_dir(out_dir):
    # if os.path.exists(out_dir):
    # logging.warning(
    #     'The output folder {} exists. New output may overwrite the old output.'.format(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    return


def config_device(cpu_only: bool):
    if not cpu_only:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            logging.warning('CUDA device is not available. Will use CPU')
    else:
        use_cuda = False
    _device = torch.device("cuda:0" if use_cuda else "cpu")
    return _device


def config_logging(log_dir, log_file_name, level=logging.WARNING, screen=True):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file_name)
    _handlers = [logging.FileHandler(log_path)]
    if screen:
        _handlers.append(logging.StreamHandler())
    logging.basicConfig(level=level, handlers=_handlers)


#
def pil_imshow(image, rectangle: list = None, multiple_rectangles=False, window_name="debug", wait_key=False,
               xyx2y2=False):
    import cv2
    import numpy as np

    for i in range(10):

        cv2_img = np.array(image)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        if rectangle is not None:
            if not multiple_rectangles:
                bbox = rectangle
                if xyx2y2:
                    cv2.rectangle(cv2_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  color=(255, 255, 0),
                                  thickness=2)
                else:
                    cv2.rectangle(cv2_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(255, 255, 0),
                                  thickness=2)
            else:
                for j in range(rectangle.shape[0]):
                    if xyx2y2:
                        pt1 = rectangle[j][0], rectangle[j][1]
                        pt2 = rectangle[j][2], rectangle[j][3]
                        cv2.rectangle(cv2_img, pt1, pt2, color=(255, 0, 0), thickness=2)
                    else:
                        pt1 = rectangle[j][0], rectangle[j][1]
                        pt2 = pt1[0] + rectangle[j][2], pt1[1] + rectangle[j][3]
                        cv2.rectangle(cv2_img, pt1, pt2, color=(255, 0, 0), thickness=2)

        cv2.imshow(window_name, cv2_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

    while True and wait_key:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return


def load_class_names_from_file(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def init_conv_layer_randomly(m):
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    if m.bias is not None:
        torch.nn.init.constant_(m.bias.data, 0.0)


def init_bn_layer_randomly(m):
    torch.nn.init.constant_(m.weight, 1.0)
    torch.nn.init.constant_(m.bias, 0.0)


def init_layer_randomly(m):
    if isinstance(m, nn.Conv2d):
        init_conv_layer_randomly(m)
    elif isinstance(m, nn.BatchNorm2d):
        init_bn_layer_randomly(m)
    else:
        pass


def untransform_bboxes(bboxes, scale, padding):
    """transform the bounding box from the scaled image back to the unscaled image."""
    x = bboxes[..., 0]
    y = bboxes[..., 1]
    w = bboxes[..., 2]
    h = bboxes[..., 3]
    # x, y, w, h = bbs
    x /= scale
    y /= scale
    w /= scale
    h /= scale
    x -= padding[0]
    y -= padding[1]
    return bboxes


def transform_bboxes(bb, scale, padding):
    """transform the bounding box from the raw image  to the padded-then-scaled image."""
    x, y, w, h = bb
    x += padding[0]
    y += padding[1]
    x *= scale
    y *= scale
    w *= scale
    h *= scale

    return x, y, w, h


# def add_coco_empty_category(old_id):
#     """The reverse of delete_coco_empty_category."""
#     starting_idx = 1
#     new_id = old_id + starting_idx
#     for missing_id in MISSING_IDS:
#         if new_id >= missing_id:
#             new_id += 1
#         else:
#             break
#     return new_id


def cxcywh_to_xywh(bbox):
    bbox[..., 0] -= bbox[..., 2] / 2
    bbox[..., 1] -= bbox[..., 3] / 2
    return bbox


def xywh_to_cxcywh(bbox):
    bbox[..., 0] += bbox[..., 2] / 2
    bbox[..., 1] += bbox[..., 3] / 2
    return bbox


def draw_result(img, boxes, show=False, class_names=None):
    if isinstance(img, torch.Tensor):
        transform = ToPILImage()
        img = transform(img)
    draw = ImageDraw.ImageDraw(img)
    show_class = (boxes.size(1) >= 6)
    if show_class:
        assert isinstance(class_names, list)
    for box in boxes:
        x, y, w, h = box[:4]
        x2 = x + w
        y2 = y + h
        draw.rectangle([x, y, x2, y2], outline='white', width=3)
        if show_class:
            class_id = int(box[5])
            class_name = class_names[class_id]
            font_size = 20
            # TODO: fix this dirty font hack that works only on Linux
            free_font = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
            class_font = ImageFont.truetype(free_font, size=font_size)
            # \
            # ImageFont.truetype("../fonts/Roboto-Regular.ttf", font_size)
            text_size = draw.textsize(class_name, font=class_font)
            draw.rectangle([x, y - text_size[1], x + text_size[0], y], fill='white')
            draw.text([x, y - font_size], class_name, font=class_font, fill='black')
    if show:
        img.show()
    return img


