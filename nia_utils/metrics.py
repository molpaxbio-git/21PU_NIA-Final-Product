# Base: YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""
# Modified by ☘ Molpaxbio Co., Ltd.
# Author: jaesik.won@molpax.com
# Date: 10/12/2021

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from nia_utils.logger import Logger


"""
Calculation Functions
"""
def calc_ap(r, p):
    tmp_r = np.concatenate(([0.0], r, [1.0]))
    tmp_p = np.concatenate(([1.0], p, [0.0]))
    tmp_p = np.flip(np.maximum.accumulate(np.flip(tmp_p)))
    
    ap = np.trapz(tmp_p, tmp_r)
    return ap, tmp_r, tmp_p
    
def plot_ap(r, p, c, save_dir, name):
    plt.plot(r, p)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f"{save_dir}/ap_curve_{name}.png", dpi=250)
    plt.close()

def ap_per_class(stats, save_dir='.', names=(), logger: Logger=None):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    """
    
    data_id = np.array([])
    true_positive = np.array([], dtype=np.bool_)
    confidance = np.array([])
    prediction = np.array([], dtype=np.int8)
    ground_truth = np.array([], dtype=np.int8)
    number_of_targets = np.array([], dtype=np.int8)
    
    for stat in stats:
        l = len(stat[2])
        data_id = np.append(data_id, stat[0] * l)
        true_positive = np.append(true_positive, stat[1][:,0])
        confidance = np.append(confidance, stat[2])
        prediction = np.append(prediction, stat[3].type(torch.uint8))
        ground_truth = np.append(ground_truth, [int(stat[4][0]) if len(stat[4]) != 0 else 5] * l)
        number_of_targets = np.append(number_of_targets, int(stat[4][0])) if len(stat[4]) != 0 else number_of_targets

    logger.process("Calculation: Sort result by confidance")
    i = np.argsort(-confidance)
    length = i.shape[0]
    data_id, ground_truth, prediction, confidance, true_positive = data_id[i], ground_truth[i], prediction[i], confidance[i], true_positive[i]
    
    unique_classes = np.unique(ground_truth)
    nc = unique_classes.shape[0]

    aps = []
    number_of_predictions = []
    
    for c in unique_classes:
        n_labels = (ground_truth == c).sum()
        i = prediction == c
        n_preds = i.sum()
        
        number_of_predictions.append(n_preds)
        
        if n_labels == 0 or n_preds == 0:
            continue
        else:
            logger.process(f"Calculation: Calculate AP per class. class {names[int(c)]}.")
            did = data_id[i]
            gt = ground_truth[i]
            pred = prediction[i]
            conf = confidance[i]
            
            tp_accum = true_positive[i].cumsum(0)
            fp_accum = (1 - true_positive[i]).cumsum(0)
            
            p = tp_accum / (tp_accum + fp_accum)
            r = tp_accum / (tp_accum[-1] + 1e-16)
            
            ap, tmp_r, tmp_p = calc_ap(r, p)
            aps.append(ap)
            plot_ap(tmp_r, tmp_p, int(c), save_dir, names[int(c)])
            
            gtt = [names[int(x.item())] for x in gt]
            predd = [names[int(x.item())] for x in pred]
            
            df = pd.DataFrame(np.vstack((did, gtt, predd, conf, tp_accum, fp_accum, p, r)).T)
            df.index + 1
            df.index.name = "No."
            df.columns = ['Data ID', 'Class - Ground Truth', 'Class - Predict', 'Confidence level', '누적 TP', '누적 FP', 'Precision', 'Recall']
            df.to_csv(f"{save_dir}/result_{names[int(c)]}.csv", encoding='utf-8-sig')
            
    number_of_targets = np.bincount(number_of_targets, minlength=nc)

    return aps, unique_classes[:-1].astype('int32'), number_of_targets, number_of_predictions
