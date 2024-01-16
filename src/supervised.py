###############################################################################
# imports 

import numpy as np
import os
import pandas as pd
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

###############################################################################
# prostprocessing attention matrices for supervised learning

def post_process_attn_mats(attn_mats):
    row_sums = attn_mats.sum(-1, keepdims = True)
    col_sums = attn_mats.sum(-2, keepdims = True)
    mat_sums = attn_mats.sum((-1, -2), keepdims = True)

    avg = (row_sums * col_sums) / mat_sums

    normalised = attn_mats - avg

    attn_mats = np.concatenate([normalised, attn_mats])

    return attn_mats

###############################################################################
# 

def build_train_test_sets(opt, attn_mats):

    # get all gene names
    tmp = pd.read_csv(os.path.join(opt.data_dir, "data.csv"), index_col = 0).index
    idn_idf = {item: i for i, item in enumerate(tmp)}

    z = np.load(os.path.join(opt.data_dir, "train_z.npy"))
    y = np.load(os.path.join(opt.data_dir, "train_y.npy"))
    trainy = np.load(os.path.join(opt.data_dir, opt.train_y_file))
    train, test = pkl.load(open(os.path.join(opt.data_dir, opt.split_file), "rb"))

    train_X = list()
    train_Y = list()
    test_X = list()
    test_Y = list()

    for item in train:

        gene_pair = z[item]
        i = idn_idf[gene_pair[0]]
        j = idn_idf[gene_pair[1]]

        train_X.append(attn_mats[:, i, j])
        train_Y.append(trainy[item])

    for item in test:

        gene_pair = z[item]
        i = idn_idf[gene_pair[0]]
        j = idn_idf[gene_pair[1]]

        test_X.append(attn_mats[:, i, j])
        test_Y.append(y[item])

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)

    return train_X, train_Y, test_X, test_Y

def fit_forests(opt, X, Y):

    preds = list()

    for i in range(opt.tree_num):
        f = RandomForestClassifier(random_state = i)
        f.fit(X, Y)
        pred = f.predict_proba(X)[:, 1]
        
        preds.append(pred)

    # convert to dataframe
    preds = pd.DataFrame(preds)

    return preds

def filter_supervision_data(preds, X, Y):
    pos = np.nonzero(np.array(Y > 0.5))[0]
    neg = np.nonzero(np.array(Y < 0.5))[0]

    cutoff1 = preds[pos].mean().mean() - preds[pos].mean().std() * 2
    cutoff2 = preds[neg].mean().mean() + preds[neg].mean().std() * 2
    cutoff3 = preds[pos].std().mean() + preds[pos].std().std() * 2
    cutoff4 = preds[neg].std().mean() + preds[neg].std().std() * 2

    low_confidence_pos = np.nonzero(np.array(preds[pos].mean() < cutoff1))[0]
    low_consistency_pos = np.nonzero(np.array(preds[pos].std() > cutoff3))[0]
    del_pos = pos[list(set(low_confidence_pos) | set(low_consistency_pos))]

    low_confidence_neg = np.nonzero(np.array(preds[neg].mean() > cutoff2))[0]
    low_consistency_neg = np.nonzero(np.array(preds[neg].std() > cutoff4))[0]
    del_neg = neg[list(set(low_confidence_neg) | set(low_consistency_neg))]

    nondelete = list(set(range(len(Y)))-set(del_pos) - set(del_neg)) # 897 gene indexes

    X = X[nondelete]
    Y = Y[nondelete]

    return X, Y

def score_fit(f, X, Y):

    pred = f.predict_proba(X)[:, 1]

    test_auc = roc_auc_score(Y, pred)
    test_auprc_ratio = average_precision_score(Y, pred) / np.mean(Y)
    n = sum(Y)

    test_epr = pd.DataFrame([pred, Y]).T.sample(frac=1).sort_values(0, ascending = False).iloc[:n][1].sum() / (
            n ** 2 / len(Y))
    return test_auc, test_auprc_ratio, test_epr
