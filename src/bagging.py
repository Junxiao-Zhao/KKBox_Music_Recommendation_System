import numpy as np
from sklearn.metrics import roc_auc_score


def get_prediction(model_ls, model_input):

    pred_dict = {}
    for i, model in enumerate(model_ls):
        pred_val = model.predict(model_input, batch_size=8192).reshape(-1)
        pred_dict[f"pred{i}"] = pred_val

    return pred_dict


def get_auc(pred_dict, target):

    auc_dict = {}
    for k, v in pred_dict.items():
        auc_dict[k] = roc_auc_score(target, v)
    auc_dict['bagging'] = roc_auc_score(
        target, np.mean(list(pred_dict.values()), axis=0))

    return auc_dict
