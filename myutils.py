import torch
import random
import yaml
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

PROJECT_FILE = os.path.split(os.path.realpath(__file__))[0]
CONFIG_FILE = os.path.join(PROJECT_FILE, 'configs.yml')

def get_params(param_type):
    f = open(CONFIG_FILE)
    params = yaml.load(f)
    return params[param_type]


def TPR_FPR(y_prob, y_true, thres, verbose=True): 
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.where(y_prob >= thres, 1, 0)

    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    fpr = (fp / (fp + tn + 1e-10))
    tpr = (tp / (tp + fn + 1e-10))
    
    if verbose:
        print('TPR:', tpr, 'FPR:', fpr,)
        # print('TN:', tn, 'TP:', tp, 'FP:', fp, 'FN:', fn)
        
    return tpr, fpr


def multi_fpr_tpr(y_prob, y_true, thres_max, thres_min=0, split = 1000, is_P_mal=True): 
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fpr = []
    tpr = []

    thresholds = np.linspace(thres_min, thres_max, split)
    for threshold in thresholds:
        if is_P_mal: 
            y_pred = np.where(y_prob >= threshold, 1, 0)
        else:
            y_pred = np.where(y_prob <= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return fpr, tpr


def multi_metrics(probs, 
                  labels,
                  thres_max=1.,
                  thres_min=0.,
                  split=1000,
                  is_P_mal=True,
                  condition=None,
                #   plot_file=None, 
                  plot_file='FRONTEND',
                  ):
    
    fprs, tprs = multi_fpr_tpr(probs, labels, thres_max, thres_min=thres_min, split=split, is_P_mal=is_P_mal)
    roc_auc = metrics.auc(fprs, tprs)
    print('roc_auc:',roc_auc)
    
    if plot_file:
        plt.figure()
        plt.plot(fprs, tprs)
        plt.title('Receiver Operating Characteristic')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        if plot_file == 'FRONTEND':
            plt.show()
        else:
            plt.savefig(plot_file)
        
    if condition is not None:
        fprs, tprs = np.asarray(fprs), np.asarray(tprs)
        if 'tpr' in condition:
            print('fpr: %.4f'%np.min(fprs[tprs>=condition['tpr']]) , '(@tpr %.4f)'%condition['tpr'])
        if 'fpr' in condition:
            print('tpr: %.4f'%np.max(tprs[fprs<=condition['fpr']]) , '(@fpr %.4f)'%condition['fpr'])
            
    # return fprs, tprs

def set_random_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False