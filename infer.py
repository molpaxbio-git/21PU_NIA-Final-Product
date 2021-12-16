# Base source author: YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset
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
from pathlib import Path

import numpy as np
import torch
import yaml

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, \
    check_suffix, check_yaml, box_iou, non_max_suppression, scale_coords, xywh2xyxy
from utils.torch_utils import select_device

from nia_utils.datasets import S3Dataset, create_dataloader
from nia_utils.metrics import ap_per_class
from nia_utils.logger import Logger

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 1]), for 1 IoU level (0.5)
    """
    # correct Array[N, 1]
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.5,  # NMS IoU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        verbose=False,  # verbose output
        project='runs/val',  # save to project/name
        name='',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        logger: Logger=None
        ):
    training = model is not None
    
    # Setup save directories
    save_dir = Path(project) / name if not training else save_dir
    if training:
        logger.setname('val')
        logger.setverbose(False)
    else:
        logger = Logger(save_dir, 'val', mode='a')
    
    # [Log] Start
    start_t = datetime.now(timezone('Asia/Seoul')).strftime("Validation started at %Y/%m/%d, %H:%M:%S")
    logger.log(start_t)
    
    # [Process] Select device
    logger.process("Select device")
    if training:
        device = next(model.parameters()).device
    else:
        device = select_device(device, batch_size=batch_size)
    
        # [Process] Load model
        logger.process("Load model")
        check_suffix(weights, '.pt')
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # [Process] Access S3
        logger.process("Accessing S3")
        dataset = S3Dataset(logger=logger)
        dataset.attempt_get_object_keys(data)

    # [Process] Half precision
    logger.process("Half precision")
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # [Process] Configuration
    logger.process("Configuration")
    model.eval()
    iouv = torch.tensor([0.5]).to(device)
    niou = iouv.numel()

    if not training:
        # [Process] Setup dataloader
        logger.process("Setup dataloader")
        dataloader = create_dataloader(dataset.pathdict['val'], dataset.bucket, dataset.names, imgsz, batch_size, gs, pad=0.5, prefix='val: ')[0]

    # [Process] Prediction: Init
    logger.process("Prediction: Initialization")
    seen = 0
    s = ('%25s' + '%11s' * 3) % ('Class', 'Labels', 'Preds', 'mAP@.5')
    stats = []
    names = {k: v for k, v in enumerate(model.names)}
    
    # [Process] Prediction: Run
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # normalization: 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run model
        logger.process("Prediction: Run model")
        out, _ = model(img)  # inference and training outputs

        # Run NMS
        logger.process("Prediction: Run NMS")
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        out = non_max_suppression(out, conf_thres, iou_thres)

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1
            logger.process("Prediction: Statistic per image " + path.stem  + '.jpg')

            # No pred && Label exists.
            if len(pred) == 0:
                if nl:
                    stats.append(([path.stem + '.jpg'], torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append(([path.stem + '.jpg'], correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)) # (correct, conf, pcls, tcls)
    
    # [Process] Calculate AP, mAP
    ap, ap_class, nt, npred = ap_per_class(stats, save_dir=save_dir, names=names, logger=logger)
    logger.process("Calculation: Calculate mAP")
    map50 = np.mean(ap)

    if training:
        logger.setverbose(True)
    
    results = {}
    # [Process] Print results
    logger.process("Print results")
    logger.result(s)
    
    pf = '%25s' + '%11i' * 2 + '%11.3g'  # print format
    logger.result(pf % ('all', nt.sum(), np.sum(npred), map50))
    results['All'] = {'Number of Targets': str(nt.sum()),
                      'Number of Predictions': str(np.sum(npred)),
                      'mAP@.5': str(map50)}

    # [Process] Print results per class
    if verbose or not training:
        for i, c in enumerate(ap_class):
            logger.result(pf % (names[c], nt[i], npred[i], ap[i]))
            results[names[c]] = {'Number of Targets': str(nt[i]),
                                 'Number of Predictions': str(npred[i]),
                                 'AP@.5': str(ap[i])}
    
    model.float()  # for training
    # [Process] Save results
    if verbose or not training:
        logger.process("Save results")
        with open(f'{save_dir}/results.yaml', 'w') as f:
            yaml.dump(results, f)
        logger.log(f"Results saved at {save_dir}")
        
    end_t = datetime.now(timezone('Asia/Seoul')).strftime("Validation ended at: %Y/%m/%d, %H:%M:%S")
    logger.log(end_t)
    
    if not training:
        logger.close()
    else:
        logger.setname('training')

    return map50


def parse_opt():
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='./nia_utils/data.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--project', default='runs/infer', help='save to project/name')
    parser.add_argument('--name', default='', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    return opt


def main(opt):
    check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=('tensorboard', 'thop'))

    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
