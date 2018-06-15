import pandas as pd
import numpy as np
import simplejson as json
import urllib


def anonymize_mindcontrol_data(data_path):
    """
    Function that takes data downloaded from Mindcontrol and removes any email
    addresses and removes datetime stamps.

    Parameters
    ----------
    data_path : string.
        This is the path to the mindcontrol data

    Returns
    -------
    anon_path : string
        path to anonymized data
    """
    assert (data_path.endswith(".json")), "please give a path to the a .json"
    assert isinstance(data_path, str), "please give a string path to the file you \
want to anonymize"
    assert os.path.exists(data_path), "the file you are pointing to does not \
exist"

    with open(data_path, 'r') as f:
        data = [eval(l) for l in f.readlines()]

    for i, d in enumerate(data):
        if "checkedBy" in d.keys():
            data[i]["checkedBy"] = data[i]["checkedBy"].split("@")[0]
            data[i].pop("checkedAt")
            if "quality_vote" in d.keys():
                for j, v in enumerate(d["quality_vote"]):
                    res = data[i]["quality_vote"][j]["checkedBy"].split("@")[0]
                    data[i]["quality_vote"][j]["checkedBy"] = res
                    data[i]["quality_vote"][j].pop("checkedAt")

    anon_path = data_path.replace(".json", "_anon.json")
    with open(anon_path, "w") as out:
        out.write(json.dumps(data))


def tidy_df(mind_df):
    mind_df_tidy = pd.DataFrame()

    for i, row in mind_df.iterrows():
        for entry in row.quality_vote:
            entry['name'] = row['name']
            try:
                entry['vote'] = int(entry['quality_check']['QC'])
                entry['notes'] = entry['quality_check']['notes_QC']
                # some people used emails as username
                entry['checkedBy'] = entry['checkedBy'].split('@')[0]
                mind_df_tidy = mind_df_tidy.append(entry, ignore_index=True)
            except KeyError:
                pass

    def calc_score(index):
        vote = mind_df_tidy.loc[index].vote
        conf = mind_df_tidy.loc[index].confidence
        if vote:
            return abs(conf)
        else:
            return -1*abs(conf)

    mind_df_tidy['score'] = mind_df_tidy.index.map(calc_score)
    return mind_df_tidy


def get_truth_labels(input_data,
                     truth_users=['anisha', 'dnkennedy',
                                  '62442katieb', 'amandae'],
                     modality="T1w",
                     threshold=4):
    """
    Function that takes anonymized data from mindcontrol and returns a
    balanced dataset of passes and fails

    Parameters
    ----------
    input_data : string.
        This is the path to the mindcontrol data
    truth_users: a list of strings
        A list of users from mindcontrol who you trust as your "ground truth"
        raters
    modality: string.
        entry_type to filter by from mindcontrol data
    threshold: int or float
        A postive number >0 and <=5 as your criteria for which images are added
        to the "gold" standard

    Returns
    -------
    log : dict
        Dictionary with "gold" standard ratings and other pertinent information
    """

    log = {}
    mind_df = pd.read_json(input_data)
    mind_df = mind_df[mind_df.entry_type == modality]
    mind_df_tidy = tidy_df(mind_df)
    log['tidy_df'] = mind_df_tidy.to_json()

    passes = mind_df_tidy[mind_df_tidy.score >= threshold]
    fails = mind_df_tidy[mind_df_tidy.score <= -threshold]

    log['passes_shape'] = passes.shape
    log['fails_shape'] = fails.shape

    # remove any images with defacing errors
    not_including = []

    for i, row in fails[fails.notes != ""].iterrows():
        if 'defac' in row.notes.lower() or 'slice' in row.notes.lower():
            not_including.append(row['name'])

    # make it unique
    not_including = list(set(not_including))

    log['not_including'] = not_including

    # filter them out from the fails
    fails = fails[~fails['name'].isin(not_including)]

    ak_pass = passes[passes.checkedBy.isin(truth_users)]  # ['name'].values
    ak_fail = fails[fails.checkedBy.isin(truth_users)]  # ['name'].values
    ak_N = min(ak_pass.shape[0], ak_fail.shape[0])

    log['truth_passes_shape'] = ak_pass.shape
    log['truth_fail_shape'] = ak_fail.shape

    np.random.seed(0)

    idx = np.arange(ak_pass.shape[0])
    np.random.shuffle(idx)
    ak_pass_subset = ak_pass.iloc[idx[:ak_N]]

    passing_names = ak_pass_subset['name'].values
    failing_names = ak_fail['name'].values

    log['passing_names'] = passing_names.tolist()
    log['failing_names'] = failing_names.tolist()

    return log
