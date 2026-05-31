from copy import deepcopy

from utils.evaluator import compute_binary_metrics, evaluate_model


def validate(y_true, y_pred):
    metrics = compute_binary_metrics(y_true, y_pred)
    return metrics["acc0"], metrics["ap"], metrics["r_acc0"], metrics["f_acc0"]


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"
    N = y_true.shape[0]

    if y_pred[0:N // 2].max() <= y_pred[N // 2:N].min():  # perfectly separable case
        return (y_pred[0:N // 2].max() + y_pred[N // 2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres


def calculate_acc_auc_f1(y_true, y_pred, thres):
    metrics = compute_binary_metrics(y_true, y_pred, thres=thres)
    return (
        metrics["r_acc0"],
        metrics["f_acc0"],
        metrics["acc0"],
        metrics["auc"],
        metrics["f1"],
        metrics["ap"],
    )


def validate_plain(model, loader):
    return evaluate_model(model, loader, predictor="binary", desc="evaluate")


def validate_multicls(model, loader):
    return evaluate_model(model, loader, predictor="softmax", desc="evaluate")


def validate_poundnet(model, loader, specific_cls=False):
    if specific_cls:
        raise NotImplementedError("specific_cls evaluation should be implemented as a Predictor.")
    return evaluate_model(model, loader, predictor="poundnet", desc="evaluate")


def validate_lnp(model_dis, data_loader):
    return evaluate_model(model_dis, data_loader, predictor="lnp", desc="evaluate")


def validate_lgrad(model_dis, data_loader):
    return evaluate_model(model_dis, data_loader, predictor="lgrad", desc="evaluate")


def validate_dire(model_dis, data_loader):
    return evaluate_model(model_dis, data_loader, predictor="dire", desc="evaluate")


def validate_dnf(model_dis, data_loader):
    return evaluate_model(model_dis, data_loader, predictor="dnf", desc="evaluate")
