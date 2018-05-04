from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import braindrAnalysis as ba

data_path = op.join(sb.__path__[0], 'data')


def test_get_truth_labels():
    """
    Testing that mindcontrol data wrangling works

    """
    # We start with actual data. We test here just that reading the data in
    # different ways ultimately generates the same arrays.
    mindcontrol_anon_data = op.join(data_path,
                                    'mindcontrol-feb-21-18_anon.json')
    ba.get_truth_labels(mindcontrol_anon_data)


def test_aggregate_braindr_votes():
    """
    Testing that braindr aggregation works
    """

    mindcontrol_anon_data = op.join(data_path,
                                    'mindcontrol-feb-21-18_anon.json')
    mc_labels = ba.get_truth_labels(mindcontrol_anon_data)

    braindr_data = op.join(data_path, 'braindr_data_2-27-18.csv')

    ba.aggregate_braindr_votes(braindr_data,
                               mc_labels['passing_names'],
                               mc_labels['failing_names'])
