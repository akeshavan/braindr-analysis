from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import simplejson as json
import os
from .due import due, Doi
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

__all__ = ["anonymize_mindcontrol_data"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Analysis for braindr",
         tags=["reference-implementation"],
         path='braindr-results')


log = {}


def model(bdr_pivot, learning_rates=[0.1], n_estimators=[200], max_depth=[2],
          test_size=0.33):
    # bdr_pivot = pd.DataFrame(braindr_pivot)
    X = bdr_pivot[[c for c in bdr_pivot.columns if c not in ['plain_average',
                                                             'truth']]].values
    y = bdr_pivot.truth.values
    log["X_shape"] = X.shape
    log['y_shape'] = y.shape

    seed = 7
    # test_size = 0.33

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=seed,
                                                        stratify=y)
    log['X_train_shape'] = X_train.shape
    # make sure everyone has a vote in the train and test
    assert(np.isfinite(X_train).sum(0).all()), 'not everyone has a vote'
    assert(np.isfinite(X_test).sum(0).all()), 'not everyone has a vote'

    model = XGBClassifier()

    # run the grid search
    param_grid = dict(learning_rate=learning_rates,
                      max_depth=max_depth,
                      n_estimators=n_estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss",
                               n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, y_train)

    # results
    log["Best: %f using %s"] = (grid_result.best_score_,
                                grid_result.best_params_)

    y_pred_prob = grid_result.predict_proba(X_test)[:, 1]
    log['y_pred_prob'] = y_pred_prob.tolist()
    log["y_test"] = y_test.tolist()

    B = grid_result.best_estimator_.get_booster()
    fscores = B.get_fscore()
    fdf = pd.DataFrame([fscores]).T.rename(columns={0: 'F'})
    not_col = ['plain_average', 'truth']
    users = [c for c in bdr_pivot.columns if c not in not_col]
    fdf['user'] = fdf.index.map(lambda x: users[int(x[1:])])
    fdf.sort_values('F', inplace=True)
    log['user_importance'] = fdf[::-1].to_json(orient='records')
    return grid_result


def aggregate_braindr_votes(braindr_data, pass_labels, fail_labels,
                            learning_rates=[0.1], n_estimators=[200],
                            max_depth=[2], test_size=0.33):
    """
    Function that aggregates braindr data using the XGBoost model

    Parameters
    ----------
    braindr_data : string.
        This is the path to the braindr data downloaded from firebase or a URL
        to the data
    pass_labels : list of strings
        a list of names that are considered passing
    fail_labels : list of strings
        a list of names that are considered failing
    learning_rates : list of floats
        a list of learning rates to grid search in XGBoost
    n_estimators : list of ints
        a list of number of estimators to grid search in XGBoost
    max_depth : list of ints
        a list of maximum tree depth for to grid search in XGBoost
    test_size : float
        fraction of data to put into test set

    Returns
    -------
    anon_path : string
        path to anonymized data
    """

    assert isinstance(braindr_data, str), "input a string path to\
braindr_data"

    if braindr_data.startswith('http'):
        braindr_df = pd.read_csv(braindr_data)
    else:
        assert(os.path.exists(braindr_data)), "please give a valid path\
 to braindr data"
        braindr_df = pd.read_table(braindr_data)

    braindr_df['subject_name'] = braindr_df.image_id.map(lambda x: x.split('__')[0])
    braindr_df_pass_subset = braindr_df[braindr_df.subject_name.isin(pass_labels)]
    braindr_df_fail_subset = braindr_df[braindr_df.subject_name.isin(fail_labels)]
    braindr_df_pass_subset['truth'] = 1
    braindr_df_fail_subset['truth'] = 0

    braindr_subset = braindr_df_pass_subset.append(braindr_df_fail_subset,
                                                   ignore_index=True)

    # count users contributions
    user_counts = braindr_subset.groupby('username')\
        .apply(lambda x: x.shape[0])
    username_keep = user_counts[user_counts >= user_counts.describe()['75%']]\
        .index.values
    bdr = braindr_subset[braindr_subset.username.isin(username_keep)]

    bdr_pivot = braindr_subset.pivot_table(columns="username",
                                           index='image_id',
                                           values='vote',
                                           aggfunc=np.mean)

    uname_img_counts = pd.DataFrame()
    for uname in bdr_pivot.columns:
        uname_img_counts.loc[uname, 'counts'] = (pd.isnull(bdr_pivot[uname]) == False).sum()

    username_keep = uname_img_counts[uname_img_counts.counts >= uname_img_counts.describe().loc['75%']['counts']]
    username_keep = username_keep.index.values

    bdr = braindr_subset[braindr_subset.username.isin(username_keep)]
    bdr_pivot = bdr.pivot_table(columns="username", index='image_id',
                                values='vote', aggfunc=np.mean)
    truth_vals = bdr.groupby('image_id').apply(lambda x: x.truth.values[0])
    bdr_pivot['truth'] = truth_vals

    plain_avg = bdr_pivot[bdr_pivot.columns[:-1]].mean(1)
    bdr_pivot['plain_average'] = plain_avg
    log['bdr_pivot'] = bdr_pivot.to_json(orient='columns')

    grid_result = model(bdr_pivot, learning_rates=learning_rates,
                        n_estimators=n_estimators, max_depth=max_depth,
                        test_size=test_size)

    modelUsers = [c for c in bdr_pivot.columns if c not in ['plain_average',
                                                            'truth']]
    braindr_full_pivot = braindr_df[braindr_df.username.isin(modelUsers)]\
        .pivot_table(columns='username', index='image_id',
                     values='vote', aggfunc=np.mean)
    # braindr_full_pivot = braindr_full_pivot[modelUsers]
    log['braindr_full_pivot_shape'] = braindr_full_pivot.shape

    X_all = braindr_full_pivot.values
    y_all_pred = grid_result.best_estimator_.predict_proba(X_all)
    # model.predict_proba(X_all)

    plain_avg = braindr_full_pivot.mean(1)
    braindr_full_pivot['average_label'] = plain_avg
    braindr_full_pivot['xgboost_label'] = y_all_pred[:, 1]

    log['output'] = braindr_full_pivot.to_json(orient='columns')
    return log  # braindr_full_pivot.to_json(orient='columns')
