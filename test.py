# Base source author: YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
"""
# Modified by ☘ Molpaxbio Co., Ltd.
# Author: jaesik won <jaesik.won@molpax.com>
# Created: 10/12/2021
# Version: 0.5
# Modified: 16/12/2021

from datetime import datetime
from pytz import timezone

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_suffix, \
    non_max_suppression, scale_coords
from utils.torch_utils import select_device

from nia_utils.datasets import S3Dataset, LoadImagesFromS3
from nia_utils.logger import Logger

@torch.no_grad()
def run(data,
        weights='yolov5s.pt',  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        ):
    # Setup save directories
    save_dir = Path(project) / name
    logger = Logger(save_dir, 'test')

    # [Log] Start
    start_t = datetime.now(timezone('Asia/Seoul')).strftime("Test started at %Y/%m/%d, %H:%M:%S")
    logger.log(start_t)
    
    # Initialize
    logger.process("Select device")
    device = select_device(device)
    
    # Load model
    logger.process("Load model")
    check_suffix(weights, '.pt')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    logger.process("Accessing S3")
    dataset = S3Dataset(logger=logger)
    dataset.attempt_get_object_keys(data)
    
    # half precision only supported on CUDA
    logger.process("Half precision")
    half &= device.type != 'cpu'  
    if half:
        model.half()  # to FP16
    
    logger.process("Setup dataset")
    dataset = LoadImagesFromS3(dataset.bucket, dataset.pathdict['test']['image'], img_size=224, stride=32)

    for path, img, im0shape, s in dataset:
        logger.process(f"Prediction| {s}")
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        # Inference
        pred, _ = model(img)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image   
            p = Path(path)  # to Path
            txt_path = str(save_dir / p.stem) # img.txt
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    line = (cls, *xyxy, conf)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(("%d %d %d %d %d %g") % line + '\n')

    logger.log(f"Results saved to {save_dir}")
    logger.close()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='nia_utils/data.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/test', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
