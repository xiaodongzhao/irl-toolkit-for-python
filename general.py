# -*- coding: utf-8 -*-

import numpy as np
import importlib


# %%------- custom functions ----------------
def method_from_mdp(mdp_model, method):
    mdp = importlib.import_module({'standardmdp': 'StandardMDP', 'linearmdp': 'LinearMDP'}[mdp_model])
    return getattr(mdp, method)


def filldefaultparams(params, defaults):
    """
    Fill in default parameters of a structure.
    :param params:
    :param defaults:
    :return:
    """
    # Get default field names.
    defaultfields = defaults.keys()
    # Step over all fields in the defaults structure
    for i in range(0, len(defaultfields)):
        if defaultfields[i] not in params:
            params[defaultfields[i]] = defaults[defaultfields]
    return params


# %%---------- translated functions ---------------------------------
def setdefaulttransferparams(test_params):
    """
    Set default general parameters for transfer test.
    :param test_params: 
    :return: 
    """
    # Create default parameters.
    default_params = {
        'verbosity': 1,
        'test_models': [['standardmdp', 'linearmdp']],
        'test_metrics': [
            ['misprediction', 'policydist', 'featexp', 'value', 'reward', 'rewarddemean', 'rewardmomentmatch']]
    }

    # Set parameters.
    test_params = filldefaultparams(test_params, default_params);
    return test_params


def setdefaulttestparams(test_params):
    """
    Set default general parameters for the test.
    :param test_params:
    :return:
    """
    # Create default parameters.
    default_params = {
        'verbosity': 2,
        'training_samples': 32,
        'training_sample_lengths': 100,
        'true_features': [],
        'true_examples': [],
        'test_models': [['standardmdp', 'linearmdp']],
        'test_metrics': [['misprediction', 'policydist', 'featexp', 'value',
                          'featvar', 'reward', 'rewarddemean', 'rewardmomentmatch']]
    }

    # Set parameters.
    test_params = filldefaultparams(test_params, default_params)
    return test_params


def printresult(test_result):
    """
    Print pre-computed IRL test result.
    :param test_result:
    :return:
    """
    for o in range(0, len(test_result)):
        if len(test_result) != 1:
            if o == 0:
                print('Printing results for processed version:')
            else:
                print('Printing results for non-processed version:')
        for i in range(0, len(test_result[o].test_models)):
            for j in range(0, len(test_result[o].test_metrics)):
                print('%s on %s, %s %s: ', test_result[o].algorithm, test_result[o].mdp,
                      test_result[o].test_models[i], test_result[o].test_metrics[j])
                metric = test_result[o].metric_scores[i, j]  # TODO: metric_scores structure
                if len(metric) == 1:
                    print('%f', metric)
                elif len(metric) == 2:
                    print('%f (%f)', metric[1], metric[2])
                else:
                    print('%f (%f vs %f)', metric[1], metric[2], metric[3])


def sampleexamples(mdp_model, mdp_data, mdp_solution, test_params):
    """
    Sample example tranjectories from the state space of a given MDP.
    :param mdp_model:
    :param mdp_data:
    :param mdp_solution:
    :param test_params:
    :return:
    """

    # Allocate training samples.
    N = test_params.training_samples
    T = test_params.training_sample_lengths
    example_samples = []

    # Sample trajectories.
    for i in range(0, N):
        # Sample initial state.
        s = np.ceil(np.random.rand(1, 1) * mdp_data.states)

        # Run sample trajectory.
        for t in range(0, T):
            # Compute optimal action for current state.
            a = method_from_mdp(mdp_model, 'action')(mdp_data, mdp_solution, s)

            # Store example. TODO
            example_samples[i, t] = [s, a]

            # Move on to next state.
            s = method_from_mdp(mdp_model, 'step')(mdp_data, mdp_solution, s, a)
    return example_samples


def evaluateirl(irl_result, true_r, example_samples, mdp_data, mdp_params, mdp_solution, mdp, _, test_models,
                test_metrics, feature_data, true_feature_map):
    """
    TODO
    :param irl_result:
    :param true_r:
    :param example_samples:
    :param mdp_data:
    :param mdp_params:
    :param mdp_solution:
    :param mdp:
    :param _:
    :param test_models:
    :param test_metrics:
    :param feature_data:
    :param true_feature_map:
    :return:
    """
    return []


def runtest(algorithm, algorithm_params, mdp_model, mdp, mdp_params, test_params):
    """
    Run IRL test with specified algorithm and example.

    :param algorithm: string specifying the IRL algorithm to use; one of:
        firl - NIPS 2010 FIRL algorithm
        bfirl - Bayesian FIRL algorithm
    :param algorithm_params: parameters of the specified algorithm
        FIRL:
            seed (0) - initialization for random seed
            iterations (10) - number of FIRL iterations to take
            depth_step (1) - increase in depth per iteration
            init_depth (0) - initial depth
        BFIRL:
            seed (0) - initialization for random seed
    :param mdp_model: string specifying MDP model to use for examples
        standardmdp - standard MDP model
    :param mdp: string specifying example to test on:
        gridworld
    :param mdp_params: mdp_params - string specifying parameters for example
      Gridworld:
        seed (0) - initialization for random seed
        n (32) - number of cells along each axis
        b (4) - size of macro cells
        determinism (1.0) - probability of correct transition
        discount (0.9) - temporal discount factor to use
    :param test_params:  general parameters for the test
        test_models - models to test on
        test_metrics - metrics to use during testing
        training_samples (32) - number of example trajectories to query
        training_sample_lengths (100) - length of each sample trajectory
        true_features ([]) - alternative set of true features
    :return: test_result - structure that contains results of the test: see evaluateirl.m
    """

    # Set default test parameters.
    test_params = setdefaulttestparams(test_params)
    # Construct MDP and features,
    [mdp_data, r, feature_data, true_feature_map] = method_from_mdp(mdp + 'build')(mdp_params)
    if test_params.true_features:
        true_feature_map = test_params.true_features

    # Solve example.
    mdp_solution = method_from_mdp(mdp_model + 'solve')(mdp_data, r)

    # Sample example trajectories.
    if not test_params.true_examples:
        example_samples = sampleexamples(mdp_model, mdp_data, mdp_solution, test_params)
    else:
        example_samples = test_params.true_examples

    # Run IRL algorithm.
    irl_result = getattr(mdp_module, algorithm + 'run')(algorithm_params, mdp_data, mdp_model, feature_data,
                                                        example_samples, true_feature_map, test_params.verbosity)

    # Evaluate result.
    test_result = evaluateirl(irl_result, r, example_samples, mdp_data, mdp_params,
                              mdp_solution, mdp, mdp_model, test_params.test_models, test_params.test_metrics,
                              feature_data, true_feature_map)
    test_result.algorithm = algorithm
    return test_result
