# Base source author: YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset
"""
# Modified by ☘ Molpaxbio Co., Ltd.
# Author: jaesik won <jaesik.won@molpax.com>
# Created: 10/12/2021
# Version: 0.5
# Modified: 16/12/2021

from datetime import datetime
from pytz import timezone

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import infer
from models.yolo import Model
from utils.general import labels_to_class_weights, init_seeds, \
    strip_optimizer, check_git_status, check_img_size, check_requirements, \
    check_yaml, check_suffix, one_cycle
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first
from utils.metrics import fitness

from nia_utils.datasets import S3Dataset, create_dataloader
from nia_utils.logger import Logger

def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          logger
          ):
    save_dir, epochs, batch_size, weights, data, cfg, workers, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.data, opt.cfg, opt.workers

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    
    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    logger.log('hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Config
    cuda = device.type != 'cpu'
    init_seeds(0)

    # Access S3
    logger.process('Accessing S3')
    dataset = S3Dataset(logger=logger)
    dataset.attempt_get_object_keys(data)
    train_dict = dataset.pathdict['train']
    val_dict = dataset.pathdict['val']
    bucket = dataset.bucket
    nc = dataset.nc
    names = list(dataset.names.keys())[:nc+1]
    dnames = dataset.names
    
    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(-1):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        logger.log(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    logger.log(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    logger.log(f"{'optimizer:'} {type(optimizer).__name__} with parameter groups " +
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    # Scheduler
    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model)

    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            logger.log(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Trainloader
    logger.process('Loading Train data')
    train_loader, dataset = create_dataloader(train_dict, bucket, dnames, imgsz, batch_size, gs,
                                              hyp=hyp, augment=True, workers=workers, prefix='train: ')
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc}. Possible class labels are 0-{nc - 1}'

    # Process 0
    logger.process('Loading Val data')
    val_loader = create_dataloader(val_dict, bucket, dnames, imgsz, batch_size * 2, gs,
                                   hyp=hyp, workers=workers, pad=0.5, prefix='val: ')[0]

    labels = np.concatenate(dataset.labels, 0)
    
    # Anchors
    model.half().float()  # pre-reduce anchor precision

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    results = 0.0
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)  # init loss class
    logger.log(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {save_dir}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        logger.log(('%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            bar_desc = ('%10s' * 2 + '%10.4g' * 5) % (
                f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(bar_desc)
            # end batch ------------------------------------------------------------------------------------------------

        logger.log(bar_desc)
            
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # mAP
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop

        results = infer.run(data,
                            batch_size=batch_size * 2,
                            imgsz=imgsz,
                            model=ema.ema,
                            dataloader=val_loader,
                            save_dir=save_dir,
                            verbose=nc < 50 and final_epoch,
                            logger=logger)

        # Update best mAP
        fi = results
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        ckpt = {'epoch': epoch,
                'best_fitness': best_fitness,
                'model': deepcopy(de_parallel(model)).half(),
                'ema': deepcopy(ema.ema).half(),
                'updates': ema.updates,
                'optimizer': optimizer.state_dict()}

        # Save last, best and delete
        torch.save(ckpt, last)
        if best_fitness == fi:
            torch.save(ckpt, best)
        del ckpt

        # Stop Single-GPU
        if stopper(epoch=epoch, fitness=fi):
            break

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    logger.log(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    # Strip optimizers
    for f in last, best:
        if f.exists():
            strip_optimizer(f)  # strip optimizers
    logger.log(f"Results saved to {save_dir}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/sP5.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./nia_utils/data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    # Checks
    check_git_status()
    check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=['thop'])

    opt.cfg, opt.hyp = check_yaml(opt.cfg), check_yaml(opt.hyp)  # check YAMLs
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.save_dir = str(Path(opt.project) / opt.name)
    logger = Logger(opt.save_dir, 'training', mode='a')
    
    # [Log] Start
    start_t = datetime.now(timezone('Asia/Seoul')).strftime("Training started at %Y/%m/%d, %H:%M:%S")
    logger.log(start_t)

    # Select device
    logger.process("Select device")
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Train
    train(opt.hyp, opt, device, logger)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
