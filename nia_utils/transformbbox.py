# Written by ☘ Molpaxbio Co., Ltd.
# Author: jaesik.won@molpax.com

from .BoxMode import BoxMode

def transformbbox(bbox, modes, iw, ih):
    m1, m2 = modes
    
    e = f'transform bbox from {m1.name} to {m2.name} return value is not implemented.'
    # same
    if m1 == m2:
        return bbox
    # XYXY_ABS
    elif m1 == BoxMode.XYXY_ABS:
        x1, y1, x2, y2 = bbox
        if m2 == BoxMode.XYWH_ABS:
            w = x2 - x1
            h = y2 - y1
            return [x1, y1, w, h]
        elif m2 == BoxMode.XYXY_REL:
            x1, x2 = x1 / iw, x2 / iw
            y1, y2 = y1 / ih, y2 / ih
            return [x1, y1, x2, y2]
        elif m2 == BoxMode.XYWH_REL:
            w = x2 - x1
            h = y2 - y1
            x1 = x1 / iw
            y1 = y1 / ih
            w = w / iw
            h = h / ih
            return [x1, y1, w, h]
        else:
            print(e)
            raise NotImplementedError
    # XYWH_ABS
    elif m1 == BoxMode.XYWH_ABS:
        x1, y1, w, h = bbox
        if m2 == BoxMode.XYXY_ABS:
            x2 = x1 + w
            y2 = y1 + h
            return [x1, y1, x2, y2]
        elif m2 == BoxMode.XYXY_REL:
            x2 = x1 + w
            y2 = y1 + h
            x1, x2 = x1 / iw, x2 / iw
            y1, y2 = y1 / ih, y2 / ih
            return [x1, y1, x2, y2]
        elif m2 == BoxMode.XYWH_REL:
            x1 = x1 / iw
            y1 = y1 / ih
            w = w / iw
            h = h / ih
            return [x1, y1, w, h]
        elif m2 == BoxMode.CCWH_REL:
            x1 = (x1 + (w / 2)) / iw
            y1 = (y1 + (h / 2)) / ih
            w = w / iw
            h = h / ih
            return [x1, y1, w, h]
        else:
            print(e)
            raise NotImplementedError
    # XYXY_REL
    elif m1 == BoxMode.XYXY_REL:
        x1, y1, x2, y2 = bbox
        if m2 == BoxMode.XYXY_ABS:
            x1, x2 = x1 * iw, x2 * iw
            y1, y2 = y1 * ih, y2 * ih
            return [x1, y1, x2, y2]
        elif m2 == BoxMode.XYWH_ABS:
            w = (x2 - x1) * iw
            h = (y2 - y1) * ih
            x1 = x1 * iw
            y1 = y1 * ih
            return [x1, y1, w, h]
        elif m2 == BoxMode.XYWH_REL:
            w = x2 - x1
            h = y2 - y1
            return [x1, y1, w, h]
        else:
            print(e)
            raise NotImplementedError
    # XYWH_REL
    elif m1 == BoxMode.XYWH_REL:
        x1, y1, w, h = bbox
        if m2 == BoxMode.XYXY_ABS:
            x2 = x1 + w
            y2 = y1 + h
            x1, x2 = x1 * iw, x2 * iw
            y1, y2 = y1 * ih, y2 * ih
            return [x1, y1, x2, y2]
        elif m2 == BoxMode.XYWH_ABS:
            w = w * iw
            h = h * ih
            x1 = x1 * iw
            y1 = y1 * ih
            return [x1, y1, w, h]
        elif m2 == BoxMode.XYXY_REL:
            x2 = x1 + w
            y2 = y1 + h
            return [x1, y1, x2, y2]
        else:
            print(e)
            raise NotImplementedError
    elif m1 == BoxMode.CCWH_REL:
        cx, cy, w, h = bbox
        if m2 == BoxMode.XYXY_ABS:
            fw = (w * iw) / 2
            fh = (h * ih) / 2
            cx = cx * iw
            cy = cy * ih
            x1 = cx - fw
            x2 = cx + fw
            y1 = cy - fh
            y2 = cy + fh
            return [x1, y1, x2, y2]
        else:
            print(e)
            raise NotImplementedError
    else:
        print(e)
        raise NotImplementedError
    
#transformbbox([0,0,10,10], (BoxMode.XYXY_ABS, BoxMode.CCWH_REL), 5, 5)
