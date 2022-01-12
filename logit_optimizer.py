# the required packages are installed at the beginning of the 'replication_notebook.ipynb'
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import time


def get_log_likelihood(y_true, y_prob, weights=None):
    """Returns the (weighted) log-likelihood."""
    if weights is None:
        weights = np.ones(y_true.shape)
    y_prob = np.where(y_prob == 1, 0.9999999, y_prob)
    y_prob = np.where(y_prob == 0, 0.0000001, y_prob)
    log_likelihood_elements = y_true * np.log(y_prob)
    log_likelihood = np.sum(weights * log_likelihood_elements) / len(y_true)
    return log_likelihood


def adj_mcfadden_r2_scorer(model, null_model, x_test, y_test):
    """Returns the adjusted Mc Fadden's pseudo R² and the weighted version of it."""
    num_classes = model.classes_.shape[0]

    full_log_likelihood = get_log_likelihood(np.eye(num_classes)[y_test], model.predict_proba(x_test))
    null_log_likelihood = get_log_likelihood(np.eye(num_classes)[y_test], null_model.predict_proba(
        np.zeros(x_test.shape[1]).reshape(1, x_test.shape[1])))

    weight_factors = y_test.sum() / y_test.value_counts(sort=False)
    scaled_weight_factors = weight_factors / weight_factors.max()
    weights = np.zeros((y_test.shape[0], num_classes))
    weights[...] = scaled_weight_factors

    weighted_full_ll = get_log_likelihood(np.eye(num_classes)[y_test], model.predict_proba(x_test), weights)
    weighted_null_ll = get_log_likelihood(np.eye(num_classes)[y_test], null_model.predict_proba(
        np.zeros(x_test.shape[1]).reshape(1, x_test.shape[1])), weights)

    # original adjusted McFadden's pseudo R². Has the issue that it maximizes for num_coef=0.
    if np.sum(model.coef_ != 0) == 0:
        adj_pseudo_r2 = -np.inf
        weighted_adj_pseudo_r2 = -np.inf
    else:
        adj_pseudo_r2 = 1 - ((full_log_likelihood - np.sum(model.coef_ != 0)) / null_log_likelihood)
        weighted_adj_pseudo_r2 = 1 - ((weighted_full_ll - np.sum(model.coef_ != 0)) / weighted_null_ll)

    return adj_pseudo_r2, weighted_adj_pseudo_r2


def cross_validation(model, null_model, X, y, folds=5):
    """Runs custom cross validation around the pseudo R² scorer."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    adj_pR2 = np.zeros(folds)
    weighted_adj_pR2 = np.zeros(folds)
    num_coef = np.zeros(folds)
    for i, [train_index, val_index] in enumerate(kf.split(y)):
        x_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        x_val = X.iloc[val_index]
        y_val = y.iloc[val_index]
        model.fit(x_train, y_train)
        num_coef[i] = np.sum(model.coef_ != 0)
        null_model.fit(X=np.zeros(x_train.shape), y=y_train)
        adj_pR2[i], weighted_adj_pR2[i] = adj_mcfadden_r2_scorer(model, null_model, x_val, y_val)
    coef_mean = num_coef.mean()
    adj_pR2_mean = adj_pR2.mean()
    weighted_adj_pR2_mean = weighted_adj_pR2.mean()
    return adj_pR2_mean, weighted_adj_pR2_mean, coef_mean


def cv_wrapper(X, y, c, l1, num_iter):
    """Runs the CV experiment and returns the result DataFrame."""
    columns = ["c", "l1", "adj_pR2", "adj_weighted_pR2", "num_coef"]
    model = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=num_iter, C=c,
                               l1_ratio=l1, fit_intercept=True, random_state=0)
    null_model = LogisticRegression(max_iter=num_iter, solver="saga", fit_intercept=True, random_state=0)
    adj_pR2, adj_weighted_pR2, num_coef = cross_validation(model, null_model, X, y, folds=3)
    result = pd.Series(data=(c, l1, adj_pR2, adj_weighted_pR2, num_coef), index=columns)
    return result


def run_grid_search(input_file, output_file):
    """runs the grid search"""
    df = pd.read_csv(input_file)
    y = df['target']
    y = y.astype('category')
    y = y.cat.reorder_categories(["none", "medium", "large", "extra_large"], ordered=True)
    y = y.cat.codes.astype(int)
    X = df.drop("target", axis=1)

    result_columns = ["c", "l1", "adj_pR2", "adj_weighted_pR2", "num_coef"]

    Carr = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    L1arr = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    iterations = 10000
    overall_start_time = time.time()
    config_tuple = []
    results = []

    for reg_strength in Carr:
        for ratio in L1arr:
            config_tuple.append([reg_strength, ratio])

    for i in range(len(config_tuple)):
        reg_strength, ratio = config_tuple[i]
        print(f'{i/len(config_tuple)*100:4.1f}% done | current parameters - strength: {reg_strength} ratio: {ratio}')
        results.append(cv_wrapper(X, y, reg_strength, ratio, iterations))
    print(f"time needed: {time.time() - overall_start_time:0.2f}s")

    result_array = np.array(results).reshape(-1, len(result_columns))
    df = pd.DataFrame(result_array, columns=result_columns)
    df.to_csv(output_file, mode='w', index=False)


if __name__ == "__main__":
    run_grid_search("data/train_data.csv", "data/logit_optim_grid_search.csv")
