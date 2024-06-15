import os
from os.path import join, isfile, isdir
import sys
sys.path.append("..")
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm
import pickle
import json


def frechet_distance(gt, pred):    
    mu1, sigma1 = gt.mean(axis=0), np.cov(gt, rowvar=False)
    mu2, sigma2 = pred.mean(axis=0), np.cov(pred, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == "__main__":
    
    ids = json.load(open("./hho_singlestage_test_seen_ids.json", "r"))
    singlestage_seen_ids = np.int32(ids["seen"])
    singlestage_unseen_ids = np.int32(ids["unseen"])
    ids = json.load(open("./stage2_exp_out_test_seen_ids.json", "r"))
    twostages_seen_ids = np.int32(ids["seen"])
    twostages_unseen_ids = np.int32(ids["unseen"])
    
    suffix = "_3"
    f_gt_train = np.load("./features_gt_train.npy")
    f_gt_test = np.load("./features_gt_test.npy")
    f_pred_singlestage = np.load("./features_pred_singlestage{}.npy".format(suffix))
    f_pred_twostages = np.load("./features_pred_twostages{}.npy".format(suffix))
    
    # merge 2 test sets
    FID = frechet_distance(f_gt_train, f_gt_test)
    print(FID)
    FID = frechet_distance(f_gt_train, f_pred_singlestage)
    print(FID)
    FID = frechet_distance(f_gt_train, f_pred_twostages)
    print(FID)
    
    # separate test sets
    print(frechet_distance(f_gt_train, f_pred_singlestage[singlestage_seen_ids[singlestage_seen_ids<f_pred_singlestage.shape[0]]]))
    print(frechet_distance(f_gt_train, f_pred_twostages[singlestage_seen_ids[singlestage_seen_ids<f_pred_twostages.shape[0]]]))
    print(frechet_distance(f_gt_train, f_pred_singlestage[singlestage_unseen_ids[singlestage_unseen_ids<f_pred_singlestage.shape[0]]]))
    print(frechet_distance(f_gt_train, f_pred_twostages[singlestage_unseen_ids[singlestage_unseen_ids<f_pred_twostages.shape[0]]]))
