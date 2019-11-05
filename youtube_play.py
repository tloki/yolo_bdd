#!/usr/bin/env python3

from models.yolov3 import load_yolov3_model
from datasets import webcam
import torch
from models.yolov3 import *
import time
from ytdl import *

cam_width = 1280
cam_height = 720

device = "cuda"

pretrained_model_path = "/Users/loki/Datasets/pretrained_models/bdd_epoch_100.pt"

model = load_yolov3_model(pretrained_model_path, device, checkpoint=True)

yt_video_url = "https://www.youtube.com/watch?v=7HaJArMDKgI"

# video_handle = get_yt_video(yt_video_url)

video_handle = "/Users/loki/Movies/video.mp4"

print(video_handle)

camera_dataset = webcam.WebCamStream(stream_url=video_handle,
                                     width=cam_width,
                                     height=cam_height,
                                     transform="default",
                                     force_res=True,
                                     transform_params=416,
                                     flip=False,
                                     wait_key=False,
                                     grayscale=False,
                                     fps=False,
                                     pause_key=False,
                                     rgb=True,
                                     skip_every=5,
                                     batch_size=16)

out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))

tm = time.time()

while True:
    tm = time.time()
    ret, imgs, tensor_imgs = camera_dataset[0]

    tensor_imgs = tensor_imgs.to(device)

    with torch.no_grad():
        detections = model(tensor_imgs)

    # car, bus truck
    detections = post_process_custom(detections, True, 0.6, 0.4, 10, [3, 4, 5])

    cxcywh_to_xywh(detections)

    detections = upscale_detections(detections, 416, 416, 1280, 720)

    print(time.time() - tm)


    for i, img in enumerate(imgs):
        frame = imshow_with_detections(img, detections[i], return_frame=True, show=False)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

out.release()



