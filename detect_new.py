# -*- coding: utf-8 -*-
# @Author   = Apexopco
# @Time     = 2021/5/7 16:54
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier

#
from utils.datasets import MyLoadImages
import os


class detect_api:
    def __init__(self):
        # absolute path to this file
        FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        # absolute path to this file's root directory
        # PARENT_DIR = os.path.join(FILE_DIR, 'weights/yolov5x6.pt')
        PARENT_DIR = os.path.join(FILE_DIR, 'weights/best.pt')
        print(PARENT_DIR)
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--weights', nargs='+', type=str, default=PARENT_DIR, help='model.pt path(s)')
        # parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
        self.parser.add_argument('--source', type=str, default=r'F:\Python\yolov5\18211.png', help='source')
        self.parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        self.parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        self.parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        self.parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.parser.add_argument('--view-img', action='store_true', help='display results')
        self.parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        self.parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        self.parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        self.parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        self.parser.add_argument('--augment', action='store_true', help='augmented inference')
        self.parser.add_argument('--update', action='store_true', help='update all models')
        self.parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        self.parser.add_argument('--name', default='exp', help='save results to project/name')
        self.parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = self.parser.parse_args()
        # weights, imgsz = self.opt.weights, self.opt.img_size
        print(self.opt)
        # check_requirements()
        # Initialize
        # set_logging()
        self.device = select_device(self.opt.device)
        print(self.opt.device, self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(self.opt.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.opt.img_size, s=self.model.stride.max())  # check img_size
        self.stride = int(self.model.stride.max())
        if self.half:
            self.model.half()  # to FP16
        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(
                self.device).eval()

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # print(self.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]


    def detect(self,source):
        weights, view_img, save_txt, imgsz =  self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        # save_dir = Path(increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok))  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # Set Dataloader
        vid_path, vid_writer = None, None
        # if webcam:
        #     view_img = True
        #     cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=imgsz)
        # else:
        save_img = True
        dataset = MyLoadImages(source, img_size=imgsz, stride=self.stride)

        # dataset = LoadImages(source, img_size=imgsz)


        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        result = []
        for img, im0s  in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            # t2 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process detections
            det = pred[0]
            im0 = im0s.copy()
            result_txt = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                    result_txt.append(line)
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_width=3)
            result.append((im0, result_txt))  # 对于每张图片，返回画完框的图片，以及该图片的标签列表。
        return result, self.names
        # print(f'Done. ({time.time() - t0:.3f}s)')

# newclass=detect_api(1,2)
# img2 = cv2.imread (r'F:\Python\yolov5\18211.png',1)
# a,b=newclass.detect([img2])
# cv2.imwrite('write1.jpg',a[0][0])
# cv2.imshow("video",a)