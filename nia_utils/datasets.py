# Base source author: YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""
# Modified by ☘ Molpaxbio Co., Ltd.
# Author: jaesik won <jaesik.won@molpax.com>
# Created: 10/12/2021
# Version: 0.5
# Modified: 16/12/2021

from pathlib import Path
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError
from PIL import Image
import numpy as np
import yaml
import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2

from utils.torch_utils import torch_distributed_zero_first
from utils.augmentations import Albumentations, letterbox, random_perspective, augment_hsv
from utils.general import xywhn2xyxy, xyxy2xywhn
from utils.datasets import InfiniteDataLoader

from nia_utils.BoxMode import BoxMode
from nia_utils.transformbbox import transformbbox
from nia_utils.logger import Logger

aws_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
aws_bucket_name = os.environ['AWS_BUCKET_NAME']


class S3Dataset:
    """
    Configuration class of S3 dataset
    """
    def __init__(self, id_=aws_key_id, secret=aws_secret_key,
                 endpoint_url='https://kr.object.ncloudstorage.com', bucket_name=aws_bucket_name, logger: Logger=None):
        self.bucket = None
        self.pathdict = {}
        self.names = {}
        self.nc = 0
        self.__access_to_S3(id_, secret, endpoint_url, bucket_name)
        self.logger = logger if isinstance(logger, Logger) else None

    def __access_to_S3(self, id_, secret, endpoint_url, bucket_name): # param: key, secret
        s3 = boto3.resource('s3',
                 aws_access_key_id=id_,
                 aws_secret_access_key=secret,
                 endpoint_url=endpoint_url)
        self.bucket = s3.Bucket(bucket_name)
    
    def __get_object_keys(self, cfg):        
        with open(cfg) as f:
            cfg = yaml.safe_load(f)
        
        assert 'nc' in cfg, "Dataset 'nc' key missing."
        
        pathdict = {}
        prefix = cfg['prefix']
        
        for kind in list(cfg['kinds']):
            pathdict[kind] = {}
            kname = cfg['kinds'][kind]
            for data in list(cfg['datas']):
                self.__logger_msg(f"Get keys from {kind}/{data}/")
                dname = cfg['datas'][data]['name']
                ext = cfg['datas'][data]['ext']
                objs = []
                for name in cfg['names']:
                    fltr = '/'.join([prefix, kname, dname, name])
                    for obj in self.bucket.objects.filter(Prefix=fltr):
                        if obj.key.find(ext) != -1:
                            objs.append(obj.key)
                pathdict[kind][data] = objs
        
        self.names = cfg['names']
        self.nc = cfg['nc']
        self.pathdict = pathdict
        
        self.__logger_msg(f"class names: {self.names}")
        self.__logger_msg(f"number of classes: {self.nc}")
        
        
    def __logger_msg(self, msg):
        self.logger.log(msg) if isinstance(self.logger, Logger) else print(msg)
        
    def attempt_get_object_keys(self, cfg):
        try:
            self.__get_object_keys(cfg)
        except ClientError as e:
            code = e.response['Error']['Code']
            print(code, end=': ')
            if code == "InvalidAccessKeyId":
                print("Access Key Id가 올바르지 않습니다.")
            elif code == "SignatureDoesNotMatch":
                print("Secret Access Key가 올바르지 않습니다.")
            print(e)


def create_dataloader(pathdict, bucket, names, imgsz, batch_size, stride, hyp=None, augment=False, pad=0.0, rect=True, rank=-1, workers=8, prefix=''):
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(pathdict, bucket, names, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      stride=int(stride),
                                      pad=pad,
                                      workers=workers,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, pathdict, bucket, names, img_size=640, batch_size=16, augment=False, hyp=None, rect=True, stride=32, workers=8, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.stride = stride
        self.pathdict = pathdict
        self.albumentations = Albumentations() if augment else None

        self.img_files = pathdict['image']
        self.label_files = pathdict['label']
        self.bucket = bucket
        self.names = names
        
        i_fs = []
        labels = []
        shapes = []
        imgs = {}
        
        func = partial(self.get_obj_from_key)
        failed_downloads = []

        # load data from s3
        with tqdm(desc="Load data from S3", total=len(self.img_files)) as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(func, ip, lp) for ip, lp in zip(self.img_files, self.label_files)]
                for future in as_completed(futures):
                    l, s, i, p = future.result()
                    i_fs.append(p)
                    labels.append(l)
                    shapes.append(s)
                    imgs[p] = i
                    if future.exception():
                        failed_downloads.append(futures[future])
                    pbar.update(1)
                    
        if len(failed_downloads) > 0:
            print("Some downloads have failed.", len(failed_downloads))
        
        self.img_files = i_fs
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.imgs = imgs

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]
            
            # Set training image shapes
            s = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    s[i] = [maxi, 1]
                elif mini > 1:
                    s[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(s) * img_size / stride + pad).astype(np.int) * stride
    
    def get_obj_from_key(self, ip, lp):
        i = self.imread_from_key(self.bucket.Object(ip))
        j = self.readjson_from_key(self.bucket.Object(lp))
        b = list(j["annotation"]["boundingbox_information"].values())
        if b[-1] == "normal skin":
            f = np.zeros((0, 5), dtype=np.float32)
        else:
            n = np.array([self.names[b[-1]]])
            l = transformbbox([int(x) for x in b[:-1]], (BoxMode.XYWH_ABS, BoxMode.CCWH_REL), i.shape[1], i.shape[0])
            f = np.array([np.concatenate([n,l])], dtype=np.float32)
        return f, i.shape[:2][::-1], i, ip
    
    def imread_from_key(self, o):
        r = o.get()
        fstream = r['Body']
        im = Image.open(fstream)
        im = np.array(im)
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    def readjson_from_key(self, o):
        r = o.get()
        f = r['Body'].read().decode('utf-8')
        return json.loads(f)

    def load_image(self, i):
        im = self.imgs[self.img_files[i]]
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]

        hyp = self.hyp

        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img, labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


class LoadImagesFromS3:  # for inference
    def __init__(self, bucket, images, img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = len(images)
        self.bucket = bucket

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        
        img0 = self.__imread_from_key(path)
        assert img0 is not None, 'Image Not Found ' + path
        s = f'image {self.count}/{self.nf}: {path} '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0.shape, s

    def __len__(self):
        return self.nf  # number of files
    
    def __imread_from_key(self, p):
        o = self.bucket.Object(p)
        r = o.get()
        fstream = r['Body']
        im = Image.open(fstream)
        im = np.array(im)
        return im