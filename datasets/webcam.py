from torch.utils.data import Dataset
import cv2
from time import time, sleep
from warnings import warn
from datasets import transforms
import torch


class FPSMeter(object):
    def __init__(self, font=cv2.FONT_HERSHEY_SIMPLEX, pos_x=None, pos_y=None, font_scale=0.5, font_color=(255, 0, 0)):
        self.font = font
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.time = time()
        self.scale = font_scale
        self.color = font_color
        self.fps = None
        self.font = font
        return

    def tick(self, img):
        width, height, _ = img.shape
        now = time()
        delta = now - self.time
        self.time = now

        self.fps = 1 // delta
        if self.pos_x is None or self.pos_y is None:
            cv2.putText(img, "FPS: " + str(int(self.fps)), (height - 76, 14), self.font,
                        self.scale, self.color, 1)
        else:
            raise NotImplementedError("fps position not implemented yet")


class WebCamStream(Dataset):

    def __init__(self, stream_url, width, height, transform="default", force_res=True, transform_params=None,
                 flip=False, wait_key=True, grayscale=False, fps=True, pause_key=False, rgb=True, skip_every=1,
                 batch_size=1):

        super(Dataset, self).__init__()

        self.capture = cv2.VideoCapture(stream_url)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.video_fps = self.capture.get(cv2.CAP_PROP_FPS)

        if rgb:
            self.capture.set(cv2.CAP_PROP_CONVERT_RGB, True)

        self.expected_shape = (height, width, 3)
        self.force_resize = force_res

        self.flip = flip
        self.wait_key = wait_key
        self.grayscale = grayscale
        self.pause_key = pause_key

        self.fps = None
        if fps:
            self.fps = FPSMeter()

        self.warned = False

        if transform == "default":
            transform = transforms.default_transform_fn_cv2(transform_params)

        self.transform = transform

        self.rgb = rgb

        self.skip_every = skip_every
        self.img_counter = 1
        self.batch_size = batch_size

    def __getitem__(self, index):
        images = []
        img_tensor = []

        for i in range(self.batch_size):

            self.capture.set(1, self.img_counter * self.skip_every)
            self.img_counter += 1
            return_code, frame = self.capture.read(1)

            if not return_code:
                    return False, None, None

            if frame is None:
                return False, None, None

            if frame.shape != self.expected_shape:

                if not self.warned:
                    warn("image{} not of expected shape{}{}".format(frame.shape, self.expected_shape,
                                                                    ", resizing!" if self.force_resize else ""))
                    self.warned = True

                if self.force_resize:
                    frame = cv2.resize(frame, (self.expected_shape[0:2][::-1]))

            if self.flip:
                frame = cv2.flip(frame, 1)

            if self.fps is not None:
                self.fps.tick(frame)

            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            transformed_bgr_image = None

            if self.transform is not None:
                transformed_bgr_image, _ = self.transform(frame)

            key = chr(cv2.waitKey(1) & 0xFF)

            if self.pause_key and key == 'p':
                while True:
                    sleep(0.3)
                    key = chr(cv2.waitKey(1) & 0xFF)
                    if key == 'p':
                        break

            if self.wait_key and key == 'q':
                print("done")
                return False, None, None


            img_tensor.append(transformed_bgr_image)
            # img_tensor.append(transformed_bgr_image)
            # if img_tensor is None:
            #     img_tensor = transformed_bgr_image
            # else:
            #     # img_tensor = torch.stack((img_tensor, transformed_bgr_image))
            #     new = torch.unsqueeze(transformed_bgr_image, 0)
            #     img_tensor = torch.stack((img_tensor, new))

            images.append(frame)

        img_tensor = torch.stack(img_tensor)
        return True, images, img_tensor



