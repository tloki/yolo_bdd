#!/usr/bin/env python3
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # train or test:
    parser.add_argument('--mode', type=str, help="'train' or 'test' the detector.")
    # data loading:
    # both:
    parser.add_argument('--dataset', dest='dataset_type', type=str, default='image_folder',
                        help="The type of the dataset used. Currently support 'coco', 'caltech' and 'image_folder'")
    parser.add_argument('--img-dir', dest='img_dir', type=str, required=True,
                        help="The path to the folder containing images to be detected or trained.")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=4,
                        help="The number of sample in one batch during training or inference.")
    parser.add_argument('--n-cpu', dest='n_cpu', type=int, default=8,
                        help="The number of cpu thread to use during batch generation.")
    parser.add_argument("--img-size", dest='img_size', type=int, default=416,
                        help="The size of the image for training or inference.")
    # training only
    parser.add_argument('--label-file', dest='label_path', type=str, default=False,
                        help="TRAINING ONLY: The path to the file of the annotations for training.",
                        )
    parser.add_argument('--no-augment', dest='data_augment', action='store_false',
                        help="TRAINING ONLY: use this option to turn off the data augmentation of the dataset."
                             "Currently only COCO dataset support data augmentation.")
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false', help="no batch shuffling")

    # model loading:
    # both:
    parser.add_argument('--weight-path', dest='weight_path', type=str, default='./weights/yolov3_original.pt',
                        help="The path to weights file for inference or finetune training.")
    parser.add_argument('--cpu-only', dest='cpu_only', action='store_true',
                        help="Use CPU only no matter whether GPU is available.")
    parser.add_argument('--from-ckpt', dest='from_ckpt', action='store_true',
                        help="Load weights from checkpoint file, where optimizer state is included.")

    # training only:
    parser.add_argument('--reset-weights', dest='reset_weights', action='store_true',
                        help="TRAINING ONLY: Reset the weights which are not fixed during training.")
    parser.add_argument('--last-n-layers', dest='n_last_layers', type=int, default=-1,
                        help="TRAINING ONLY: Unfreeze the last n layers for retraining.")
    parser.add_argument('--load-only-darknet', dest='only_darknet', action='store_true',
                        help="TRAINING ONLY: Reset the weights which are not fixed during training.")

    # logging:
    # both:
    parser.add_argument('--log-dir', dest='log_dir', type=str, default='../log',
                        help="The path to the directory of the log files.")
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Include INFO level log messages.")
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help="Include DEBUG level log messages.")

    # saving:
    # inference only:
    parser.add_argument('--n-iter', dest='n_iter', type=int, default=-1,
                        help="INFERENCE ONLY: clip number of iterations")
    parser.add_argument('--out-dir', dest='out_dir', type=str, default='../output',
                        help="INFERENCE ONLY: The path to the directory of output files.")
    parser.add_argument('--save-img', dest='save_img', action='store_true',
                        help="INFERENCE ONLY: Save output images with detections to output directory.")
    parser.add_argument('--save-det', dest='save_det', action='store_true',
                        help="INFERENCE ONLY: Save detection results in json format to output directory")
    # training only:
    parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, default="../checkpoints",
                        help="TRAINING ONLY: directory where model checkpoints are saved")
    parser.add_argument('--save-every-epoch', dest='save_every_epoch', type=int, default=1,
                        help="TRAINING ONLY: Save weights to checkpoint file every X epochs.")
    parser.add_argument('--save-every-batch', dest='save_every_batch', type=int, default=0,
                        help="TRAINING ONLY: Save weights to checkpoint file every X batches. "
                             "If value is 0, batch checkpoint will turn off.")

    # validation only
    parser.add_argument('--start-batch', dest='start', type=int, required=False, help='validation first batch offset')
    parser.add_argument('--end-batch', dest='stop', type=int, required=False, help='validation last batch offset')

    # training parameters:
    parser.add_argument('--epochs', dest='n_epoch', type=int, default=30,
                        help="TRAINING ONLY: The number of training epochs.")
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=1E-4,
                        help="TRAINING ONLY: The training learning rate.")

    # inference parameters:
    parser.add_argument('--class-path', dest='class_path', type=str, default='../data/coco.names',
                        help="TINFERENCE ONLY: he path to the file storing class label names.")
    parser.add_argument('--nms-thres', dest='nms_thres', type=float, default=0.4,
                        help="INFERENCE ONLY: iou threshold for non-maximum suppression during inference.")
    parser.add_argument('--conf-thres', dest='conf_thres', type=float, default=0.8,
                        help="INFERENCE ONLY: confidence threshold for bbox rejection.")
    _options = parser.parse_args()
    return _options
