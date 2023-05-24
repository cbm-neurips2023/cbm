import scipy
import sklearn
import numpy as np
from sklearn import svm, ensemble
from sklearn.linear_model import Lasso


def compute_sap(params, fac_train, enc_train, fac_test, enc_test):
    """
    Separated Attribute Predictability (SAP) score, https://arxiv.org/abs/1711.00848
    :param params:
    :param fac_train: (num_samples, factor_dim)
    :param enc_train: (num_samples, feature_dim)
    :param fac_test: (num_samples, factor_dim)
    :param enc_test: (num_samples, feature_dim)
    :return:
    """
    fac = np.concatenate([fac_train, fac_test])
    enc = np.concatenate([enc_train, enc_test])

    fac_dim = fac.shape[1]
    enc_dim = enc.shape[1]
    score_matrix = np.zeros([fac_dim, enc_dim])

    for i, fac_inner_dim_i in zip(range(fac_dim), params.fac_inner_dim):
        for j in range(enc_dim):
            if not params.continuous_state and fac_inner_dim_i > 1:
                fac_train_i, fac_test_i = fac_train[:, i], fac_test[:, i]
                enc_train_j, enc_test_j = enc_train[:, i], enc_test[:, i]
                classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                classifier.fit(fac_train_i[:, np.newaxis], enc_train_j)
                pred = classifier.predict(fac_test_i[:, np.newaxis])
                score_matrix[i, j] = np.mean(pred == enc_test_j)
            else:
                fac_i = fac[:, i]
                enc_j = enc[:, j]
                cov_i_j = np.cov(enc_j, fac_i, ddof=1)
                cov_enc_fac = cov_i_j[0, 1] ** 2
                var_enc = cov_i_j[0, 0]
                var_fac = cov_i_j[1, 1]
                if var_enc > 1e-12:
                    score_matrix[i, j] = cov_enc_fac / (var_enc * var_fac)
                else:
                    score_matrix[i, j] = 0

    sorted_matrix = np.sort(score_matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])


def compute_dci(params, fac_train, enc_train, fac_test, enc_test):
    """
    Disentanglement Completeness Informativeness (DCI) disentanglement, https://openreview.net/forum?id=By-7dz-AZ
    :param params:
    :param fac_train: (num_samples, factor_dim)
    :param enc_train: (num_samples, feature_dim)
    :param fac_test: (num_samples, factor_dim)
    :param enc_test: (num_samples, feature_dim)
    :return:
    """

    num_train, fac_dim = fac_train.shape
    enc_dim = enc_train.shape[1]
    importance_matrix = np.zeros([enc_dim, fac_dim])

    fac = np.concatenate([fac_train, fac_test])
    enc = np.concatenate([enc_train, enc_test])
    normalized_enc = (enc - enc.mean(axis=0)) / enc.std(axis=0)
    normalized_enc_train, normalized_enc_test = normalized_enc[:num_train], normalized_enc[num_train:]

    for i, fac_inner_dim_i in zip(range(fac_dim), params.fac_inner_dim):
        fac_train_i = fac_train[:, i]
        if not params.continuous_factor and fac_inner_dim_i > 1:
            model = ensemble.GradientBoostingClassifier()
            model.fit(enc_train, fac_train_i)
            importance_matrix[:, i] = np.abs(model.feature_importances_)
        else:
            normalized_fac_i = (fac[:, i] - fac[:, i].mean()) / fac[:, i].std()
            normalized_fac_train_i, normalized_fac_test_i = normalized_fac_i[:num_train], normalized_fac_i[num_train:]
            min_test_error = np.inf
            for alpha in np.linspace(0, 5, 51):
                model = Lasso(alpha=alpha)
                model.fit(normalized_enc_train, normalized_fac_train_i)
                test_error = np.abs(model.predict(normalized_enc_test) - normalized_fac_test_i).mean()
                if test_error < min_test_error:
                    min_test_error = test_error
                    importance_matrix[:, i] = np.abs(model.coef_)

    per_code = 1. - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def compute_mutual_information(fac, enc):
    """
    Mutual information, assuming factors are all discrete variables
    :param fac: (num_samples, factor_dim)
    :param enc: (num_samples, feature_dim)
    :return:
    """
    fac_dim = fac.shape[1]
    enc_dim = enc.shape[1]
    mutual_information = np.zeros([enc_dim, fac_dim])
    discretized_enc = np.zeros_like(enc)
    for i in range(enc_dim):
        discretized_enc[:, i] = np.digitize(enc[:, i], np.histogram(enc[:, i], 20)[1][:-1])

    for i in range(enc_dim):
        for j in range(fac_dim):
            mutual_information[i, j] = sklearn.metrics.mutual_info_score(fac[:, j], discretized_enc[:, i])
    return mutual_information


def compute_mig(params, fac_train, enc_train, fac_test, enc_test):
    """
    Mutual Information Gap, https://arxiv.org/pdf/1802.04942.pdf
    :param params:
    :param fac_train: (num_samples, factor_dim)
    :param enc_train: (num_samples, feature_dim)
    :param fac_test: (num_samples, factor_dim)
    :param enc_test: (num_samples, feature_dim)
    :return:
    """
    assert not params.continuous_factor and (params.fac_inner_dim > 1).all()

    fac = np.concatenate([fac_train, fac_test])
    enc = np.concatenate([enc_train, enc_test])
    fac_dim = fac.shape[1]

    mutual_information = compute_mutual_information(fac, enc)

    fac_entropy = np.zeros(fac_dim)
    for j in range(fac_dim):
        fac_entropy[j] = sklearn.metrics.mutual_info_score(fac[:, j], fac[:, j])

    sorted_mi = np.sort(mutual_information, axis=0)[::-1]
    mig = ((sorted_mi[0, :] - sorted_mi[1, :]) / fac_entropy).mean()
    return mig


def compute_modularity(params, fac_train, enc_train, fac_test, enc_test):
    """
    Modularity, https://arxiv.org/pdf/1802.05312.pdf
    :param params:
    :param fac_train: (num_samples, factor_dim)
    :param enc_train: (num_samples, feature_dim)
    :param fac_test: (num_samples, factor_dim)
    :param enc_test: (num_samples, feature_dim)
    :return:
    """
    assert not params.continuous_factor and (params.fac_inner_dim > 1).all()

    fac = np.concatenate([fac_train, fac_test])
    enc = np.concatenate([enc_train, enc_test])

    mutual_information = compute_mutual_information(fac, enc)

    squared_mi = np.square(mutual_information)
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] - 1.)
    delta = numerator / denominator
    modularity_score = 1 - delta
    index = (max_squared_mi == 0)
    modularity_score[index] = 0
    return np.mean(modularity_score)
