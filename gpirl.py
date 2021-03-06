# -*- coding: utf-8 -*-

# GPIRL algorithm based on irl_toolkit/GPIRL
import numpy as np
import matplotlib.pyplot as plt
from General import filldefaultparams

# Fill in default parameters for the GPIRL algorithm
# Create default parameters.
default_params = {
    'seed': 0,
    # Optional initial values.
    'initial_r': [],
    'initial_gp': [],
    # Which parameters to learn.
    'warp_x': 0,
    'learn_noise': 0,
    'learn_rbf': 1,
    # Parameters for random restarts.
    'restart_tolerance': 1e1,
    'initial_rewards': 5,
    'warp_x_restarts': 4,
    # These are hyperparameter transformations for positivity constraints.
    'ard_xform': 'exp',
    'noise_xform': 'exp',
    'rbf_xform': 'exp',
    'warp_l_xform': 'exp',
    'warp_c_xform': 'exp',
    'warp_s_xform': 'exp',
    # These are hyperparameter priors.
    'ard_prior': 'logsparsity',
    'noise_prior': 'g0',
    'rbf_prior': 'none',
    'warp_l_prior': 'g0',
    'warp_c_prior': 'gamma',
    'warp_s_prior': 'g0',
    # These are prior wights and parameters.
    'ard_prior_wt': 1,
    'noise_prior_wt': 1,
    'rbf_prior_wt': 1,
    'warp_l_prior_wt': 1,
    'warp_c_prior_wt': 0.5,
    'warp_s_prior_wt': 1,
    'gamma_shape': 2,
    # These are initial values.
    'ard_init': 1,
    'noise_init': 1e-2,
    'rbf_init': 5,
    'warp_c_init': 2.0,
    'warp_l_init': 1,
    'warp_s_init': 1,
    # These parameters control how the inducing points are selected.
    'inducing_pts': 'examplesplus',
    'inducing_pts_count': 64}


def gpirldefaultparams(algorithm_params):
    return filldefaultparams(algorithm_params, default_params)


def cartaverage(tree, feature_data):
    """
    Return average reward for given regression tree.
    :param tree:
    :param feature_data:
    :return:
    """
    if tree.type == 0:
        # Simply return the average.
        r = np.tile(tree.mean, [np.shape(feature_data.splittable)[0], 1])
    else:
        # Compute reward on each side.
        ltR = cartaverage(tree.ltTree, feature_data)
        gtR = cartaverage(tree.gtTree, feature_data)

        # Combine.
        ind = np.tile(feature_data.splittable[:, tree.test], (1, np.shape(ltR)[0]))
        r = (1 - ind).dot(ltR) + ind.dot(gtR)

    return r



