"""Original quality/cost target construction and five-fold estimators."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from source_core import kernel_rows, kr_solve, kridge_fit_predict


def build_menu(scores, costs, models, reference, thetas, primary_order=None):
    ref = models.index(reference)
    actions = [(index, None) for index in range(len(models))]
    primaries = [models.index(name) for name in primary_order] if primary_order else [index for index in range(len(models)) if index != ref]
    actions += [(index, theta) for index in primaries for theta in thetas]
    values, spend, names = [], [], []
    for index, theta in actions:
        if theta is None:
            values.append(scores[:, index]); spend.append(costs[:, index])
            names.append("direct:" + models[index])
        else:
            hit = (scores[:, index] >= theta - 1e-12).astype(float)
            values.append(hit * scores[:, index] + (1 - hit) * scores[:, ref])
            spend.append(costs[:, index] + (1 - hit) * costs[:, ref])
            names.append(f"attempt:{models[index]}@{theta:g}")
    return np.column_stack(values), np.column_stack(spend), actions, names


def targets(scores, costs, actions, profile):
    attempts = [(index, theta) for index, theta in actions if theta is not None]
    hit = np.column_stack([(scores[:, index] >= theta - 1e-12).astype(float) for index, theta in attempts])
    accepted = np.column_stack([hit[:, k] * scores[:, index] for k, (index, _) in enumerate(attempts)])
    logged = np.log(np.maximum(costs, 1e-7)) if profile == "crb" else np.log(costs)
    return np.hstack([scores, logged, hit, accepted])


def fit_predict(Xfit, Xtest, values, folds, profile):
    """Per-target alpha selection uses OOF MSE; test gets the full-fit model."""
    alphas = [0.3, 1.0, 3.0, 10.0] if profile == "crb" else [1.0, 3.0, 10.0, 30.0]
    if profile == "crb":
        Kfit = (Xfit @ Xfit.T).toarray().astype(np.float64)
        Ktest = (Xtest @ Xfit.T).toarray().astype(np.float64)
    else:
        Kfit = kernel_rows(Xfit, Xfit.T.tocsr())
        Ktest = kernel_rows(Xtest, Xfit.T.tocsr())
    oof = {alpha: np.empty_like(values) for alpha in alphas}
    for train, valid in folds:
        for alpha in alphas:
            if profile == "crb":
                predicted = kridge_fit_predict(Kfit[np.ix_(train, train)], values[train],
                                               Kfit[np.ix_(valid, train)], alpha)
            else:
                mean = values[train].mean(0)
                dual, _ = kr_solve(Kfit, train, values[train] - mean, alpha)
                predicted = Kfit[np.ix_(valid, train)] @ dual.astype(np.float32) + mean
            oof[alpha][valid] = predicted
    best = np.array([((oof[alpha] - values) ** 2).mean(0) for alpha in alphas]).argmin(0)
    held = np.column_stack([oof[alphas[index]][:, column] for column, index in enumerate(best)])
    test = np.empty((Xtest.shape[0], values.shape[1]))
    for index in sorted(set(best.tolist())):
        columns = np.where(best == index)[0]
        if profile == "crb":
            test[:, columns] = kridge_fit_predict(Kfit, values[:, columns], Ktest, alphas[index])
        else:
            mean = values[:, columns].mean(0)
            dual, _ = kr_solve(Kfit, np.arange(len(Kfit)), values[:, columns] - mean, alphas[index])
            test[:, columns] = Ktest @ dual.astype(np.float32) + mean
    return held, test, [alphas[index] for index in best], Kfit, Ktest


def compose(prediction, calibration, actions, nmodels, reference_index):
    quality = np.clip(prediction[:, :nmodels], 0, 1)
    costs = np.exp(prediction[:, nmodels:2 * nmodels]) * calibration
    natt = sum(theta is not None for _, theta in actions)
    hit = np.clip(prediction[:, 2 * nmodels:2 * nmodels + natt], 0, 1)
    accepted = np.clip(prediction[:, 2 * nmodels + natt:], 0, 1)
    values, spend, k = [], [], 0
    for model, theta in actions:
        if theta is None:
            values.append(quality[:, model]); spend.append(costs[:, model])
        else:
            values.append(accepted[:, k] + (1 - hit[:, k]) * quality[:, reference_index])
            spend.append(costs[:, model] + (1 - hit[:, k]) * costs[:, reference_index])
            k += 1
    return np.column_stack(values), np.column_stack(spend)


def knn_predictions(Kfit, Ktest, values, spend, k=20):
    leave_one_out = Kfit.copy()
    np.fill_diagonal(leave_one_out, -np.inf)
    train_neighbors = np.argpartition(-leave_one_out, k, axis=1)[:, :k]
    test_neighbors = np.argpartition(-Ktest, k, axis=1)[:, :k]
    return ((values[train_neighbors].mean(1), spend[train_neighbors].mean(1)),
            (values[test_neighbors].mean(1), spend[test_neighbors].mean(1)))


def routellm_predictions(Xfit, Xtest, scores, reference_index, weak_index, folds):
    labels = (scores[:, reference_index] > scores[:, weak_index]).astype(int)
    best = None
    for regularization in [0.3, 1.0, 3.0, 10.0]:
        held = np.empty(len(labels))
        for train, valid in folds:
            model = LogisticRegression(C=regularization, max_iter=2000).fit(Xfit[train], labels[train])
            held[valid] = model.predict_proba(Xfit[valid])[:, 1]
        auc = roc_auc_score(labels, held)
        if best is None or auc > best[0]:
            best = (auc, regularization, held)
    model = LogisticRegression(C=best[1], max_iter=2000).fit(Xfit, labels)
    return best[2], model.predict_proba(Xtest)[:, 1], {"C": best[1], "oof_auc": float(best[0])}
