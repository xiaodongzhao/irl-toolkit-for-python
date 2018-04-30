# -*- coding: utf-8 -*-


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
