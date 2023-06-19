
from src.estimator import *

n_splits = 2
indep_test_discreet = [gsq, mv_fisherz, fisherz]
indep_test_cont = [kci]
indep_test_cat = []

alpha_cons = [0.05, 0.1, 0.15]
alpha_aggr = [0.2, 0.25, 0.3]
alpha_nopref = [0.05]
out = "."

preferences = ['Лучше, если будут отсутствовать некоторые существующие связи',
                            'Лучше, если будет присутствовать связи, которых на самом деле нет',
                            'Нет предпочтений']
def get_alphas_from_prefs(prefs):
    if prefs == preferences[0]:
        alpha = alpha_cons
    elif prefs == preferences[1]:
        alpha = alpha_aggr
    else:
        alpha = alpha_nopref
    return alpha


def choose_metric(data_type):
    if data_type == 'Категориальные':
        metric_type = 'MSS'
    else:
        metric_type = 'MSE'
    return metric_type

def choose_alpha(data, alphas, test, date_type, kernel=None, polyd=None, width=None):

    if kernel != None:
        if polyd != None:
            _, best_alpha, launch_results, launch_metrics, launch_params = find_best_params(data, alphas, test, date_type, kernel=kernel, polyd=polyd)
        elif width != None:
            _, best_alpha, launch_results, launch_metrics, launch_params = find_best_params(data, alphas, test, date_type,kernel=kernel, width=width)
        else:
            _, best_alpha, launch_results, launch_metrics, launch_params = find_best_params(data, alphas, test,date_type, kernel=kernel)
    else:
        _, best_alpha, launch_results, launch_metrics, launch_params = find_best_params(data, alphas, test, date_type)

    return best_alpha, launch_results, launch_metrics, launch_params


def unpack_best_launch(launch_results, launch_metrics, metric_type):

    if metric_type == 'MSE':
        best_result = min(launch_metrics, key=launch_results.get)
    else:
        best_result = max(launch_metrics, key=launch_results.get)
    best_test = launch_results[best_result]
    best_metric = launch_metrics[best_result]

    return best_test, best_metric, best_result


def compare_to_old_alpha(new_best_launch, old_best_launch, baseline_mean, metric_type, baseline_beaten=0, new_best=0):
    if metric_type == 'MSE':
        if new_best_launch < old_best_launch:
            new_best = 1
            if new_best_launch < baseline_mean:
                baseline_beaten = 1

    else:
        if new_best_launch > old_best_launch:
            new_best = 1
            if new_best_launch > baseline_mean:
               baseline_beaten = 1


    return new_best, baseline_beaten

