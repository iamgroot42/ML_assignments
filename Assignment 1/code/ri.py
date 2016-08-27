from scipy.misc import comb
from scipy.sparse import coo_matrix
import numpy as np


def comb2(n):
    return comb(n, 2, exact=1)


def check_clusterings(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred


def contingency_matrix(labels_true, labels_pred, eps=None, max_n_classes=5000):
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    if n_classes > max_n_classes:
        raise ValueError("Too many classes for a clustering metric. If you "
                         "want to increase the limit, pass parameter "
                         "max_n_classes to the scoring function")
    if n_clusters > max_n_classes:
        raise ValueError("Too many clusters for a clustering metric. If you "
                         "want to increase the limit, pass parameter "
                         "max_n_classes to the scoring function")
    contingency = coo_matrix((np.ones(class_idx.shape[0]),
                              (class_idx, cluster_idx)),
                             shape=(n_classes, n_clusters),
                             dtype=np.int).toarray()
    if eps is not None:
        contingency = contingency + eps
    return contingency


def rand_score(labels_true, labels_pred, max_n_classes=5000):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    if (classes.shape[0] == clusters.shape[0] == 1 or
            classes.shape[0] == clusters.shape[0] == 0 or
            classes.shape[0] == clusters.shape[0] == len(labels_true)):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred,
                                     max_n_classes=max_n_classes)

    sum_comb = sum(n_ij**2 for n_ij in contingency.flatten())
    sum_comb_c = (np.asarray(contingency.sum(axis=1)) ** 2).sum()
    sum_comb_k = (np.asarray(contingency.sum(axis=0)) ** 2).sum()

    n = contingency.sum()
    a = (sum_comb - n)/2.0;
    b = (sum_comb_c - sum_comb)/2
    c = (sum_comb_k - sum_comb)/2
    d = (sum_comb + n**2 - sum_comb_c - sum_comb_k)/2
    
    return (a+d)/(a+b+c+d)
