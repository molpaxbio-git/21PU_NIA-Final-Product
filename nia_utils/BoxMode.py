# Written by ☘ Molpaxbio Co., Ltd.
# Author: jaesik.won@molpax.com

from enum import IntEnum

class BoxMode(IntEnum):
    XYXY_ABS= 0
    XYWH_ABS= 1
    XYXY_REL= 2
    XYWH_REL= 3
    CCWH_REL= 4