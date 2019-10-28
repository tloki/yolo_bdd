#!/usr/bin/env python3

import argparse
from argument_parsing import parse_args

from training import run_yolo_training
from inference import run_yolo_inference
from eval import run_yolo_evaluation


def main(args: argparse.Namespace):
    if args.mode == 'train':
        run_yolo_training(args)
    elif args.mode == 'test':
        run_yolo_inference(args)
    elif args.mode == 'eval':
        run_yolo_evaluation(args)

    else:
        raise ValueError("mode should be 'train', 'test' or 'eval'")


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
