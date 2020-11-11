import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import config
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation

plt.rcParams['figure.figsize'] = (24, 16)
plt.switch_backend('agg')
this_module = sys.modules[__name__]

parser = argparse.ArgumentParser()

parser.add_argument("--weight", type=str, default=config['initial_checkpoint'],
                    help="path to model weights to be used")
parser.add_argument("--out-dir", type=str, default=config['out_dir'],
                    help="path to save the results")
parser.add_argument("--test-set-name", type=str, default=config['test_set_name'],
                    help="path to save the results")
parser.add_argument("--prob_threshold", type=float, default=config['prob_threshold'],
                    help="threshold of the probability")

def main():
    args = parser.parse_args()
    test_set_name = args.test_set_name
    prob_threshold = args.prob_threshold

    initial_checkpoint = args.weight
    checkpoint = torch.load(initial_checkpoint)
    epoch = checkpoint['epoch']

    out_dir = args.out_dir
    save_dir = os.path.join(out_dir, 'res', str(epoch))

    # Generate prediction csv for the use of performning FROC analysis
    # Save both rpn and rcnn results
    rpn_res = []
    rcnn_res = []
    ensemble_res = []
    filenames = np.genfromtxt(test_set_name, dtype=str)
    for pid in filenames:
        if config['shift'] is not None:
            shift = np.load(config['shift'] + str(pid) + '_origin.npy')[::-1]
        else:
            shift = 0

        if os.path.exists(os.path.join(save_dir, '%s_rpns.npy' % (pid))):
            rpns = np.load(os.path.join(save_dir, '%s_rpns.npy' % (pid)))
            rpns = rpns[rpns[:,0] >= prob_threshold]
            rpns = rpns[:, [3, 2, 1, 4, 0]]
            rpns[:, [0, 1, 2]] = rpns[:, [0, 1, 2]] + shift
            names = np.array([[pid]] * len(rpns))
            rpn_res.append(np.concatenate([names, rpns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_rcnns.npy' % (pid))):
            rcnns = np.load(os.path.join(save_dir, '%s_rcnns.npy' % (pid)))
            rcnns = rcnns[rcnns[:,0] >= prob_threshold]
            rcnns = rcnns[:, [3, 2, 1, 4, 0]]
            rcnns[:, [0, 1, 2]] = rcnns[:, [0, 1, 2]] + shift
            names = np.array([[pid]] * len(rcnns))
            rcnn_res.append(np.concatenate([names, rcnns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_ensembles.npy' % (pid))):
            ensembles = np.load(os.path.join(save_dir, '%s_ensembles.npy' % (pid)))
            ensembles = ensembles[ensembles[:,0] >= prob_threshold]
            ensembles = ensembles[:, [3, 2, 1, 4, 0]]
            ensembles[:, [0, 1, 2]] = ensembles[:, [0, 1, 2]] + shift
            names = np.array([[pid]] * len(ensembles))
            ensemble_res.append(np.concatenate([names, ensembles], axis=1))
    
    rpn_res = np.concatenate(rpn_res, axis=0)
    rcnn_res = np.concatenate(rcnn_res, axis=0)
    ensemble_res = np.concatenate(ensemble_res, axis=0)
    col_names = ['seriesuid','coordX','coordY','coordZ','diameter_mm', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    ensemble_submission_path = os.path.join(eval_dir, 'submission_ensemble.csv')
    
    df = pd.DataFrame(rpn_res, columns=col_names)
    df.to_csv(rpn_submission_path, index=False)

    df = pd.DataFrame(rcnn_res, columns=col_names)
    df.to_csv(rcnn_submission_path, index=False)

    df = pd.DataFrame(ensemble_res, columns=col_names)
    df.to_csv(ensemble_submission_path, index=False)

    # Start evaluating
    if not os.path.exists(os.path.join(eval_dir, 'rpn')):
        os.makedirs(os.path.join(eval_dir, 'rpn'))
    if not os.path.exists(os.path.join(eval_dir, 'rcnn')):
        os.makedirs(os.path.join(eval_dir, 'rcnn'))
    if not os.path.exists(os.path.join(eval_dir, 'ensemble')):
        os.makedirs(os.path.join(eval_dir, 'ensemble'))
    
    #noduleCADEvaluation('evaluationScript/annotations/LIDC/3_annotation.csv',
    #'evaluationScript/annotations/LIDC/3_annotation_excluded.csv',
    #test_set_name, rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    #noduleCADEvaluation('evaluationScript/annotations/LIDC/3_annotation.csv',
    #'evaluationScript/annotations/LIDC/3_annotation_excluded.csv',
    #test_set_name, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    #noduleCADEvaluation('evaluationScript/annotations/LIDC/3_annotation.csv',
    #'evaluationScript/annotations/LIDC/3_annotation_excluded.csv',
    #test_set_name, ensemble_submission_path, os.path.join(eval_dir, 'ensemble'))

    #noduleCADEvaluation('/home/vincentwu/luna16/annotations.csv',
    #'/home/vincentwu/luna16/annotations_excluded.csv',
    #test_set_name, rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    #noduleCADEvaluation('/home/vincentwu/luna16/annotations.csv',
    #'/home/vincentwu/luna16/annotations_excluded.csv',
    #test_set_name, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    #noduleCADEvaluation('/home/vincentwu/luna16/annotations.csv',
    #'/home/vincentwu/luna16/annotations_excluded.csv',
    #test_set_name, ensemble_submission_path, os.path.join(eval_dir, 'ensemble'))

    noduleCADEvaluation('/mnt/QNAP/home/jenny/CGH_LungNodule/CGH_annotations_iou1_ALD_fixmerge.csv', None,
    test_set_name, rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    noduleCADEvaluation('/mnt/QNAP/home/jenny/CGH_LungNodule/CGH_annotations_iou1_ALD_fixmerge.csv', None,
    test_set_name, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    noduleCADEvaluation('/mnt/QNAP/home/jenny/CGH_LungNodule/CGH_annotations_iou1_ALD_fixmerge.csv', None,
    test_set_name, ensemble_submission_path, os.path.join(eval_dir, 'ensemble'))
        
    print

if __name__ == '__main__':
    main()