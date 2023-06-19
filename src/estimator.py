from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import gsq
from causallearn.utils.cit import mv_fisherz, chisq
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
from collections import defaultdict
from sklearn.model_selection import KFold
import pickle
import networkx as nx
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from causallearn.utils.cit import kci
from sklearn.metrics import matthews_corrcoef as mcc
#import crud
#import schemas

# “fisherz”: Fisher’s Z conditional independence test.

# “chisq”: Chi-squared conditional independence test.

# “gsq”: G-squared conditional independence test.

# “mv_fisherz”: Missing-value Fisher’s Z conditional independence test.
discreet = [gsq, fisherz]

def get_benchmark_cat(df, n_splits=2):
    benchmark = {}
    model = RandomForestClassifier()
    for column in df.columns:
        X, y = df.drop(column, axis=1), df[column]
        kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        kfold = kf.split(X, y)
        score = []
        for k, (train, test) in enumerate(kfold):
            model.fit(X.iloc[train, :], y.iloc[train])
            result = mcc(model.predict(X.iloc[test, :]), y.iloc[test])
            score.append(result)
        benchmark[column] = np.mean(score)
    return benchmark, np.mean(list(benchmark.values()))
def get_benchmark_cont(df, n_splits=2):
    benchmark = {}
    forest = RandomForestRegressor()
    for column in df.columns:
        X, y = df.drop(column, axis=1), df[column]
        kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        kfold = kf.split(X, y)
        score = []
        for k, (train, test) in enumerate(kfold):
            forest.fit(X.iloc[train, :], y.iloc[train])
            result = mse(forest.predict(X.iloc[test, :]), y.iloc[test])
            score.append(result)
        benchmark[column] = np.mean(score)
    return benchmark, np.mean(list(benchmark.values()))


def ind(array, item1, item2=None):
    temp1 = []
    temp2 = []
    for idx, val in np.ndenumerate(array):
        if val == item1:
            temp1.append(*idx)
        if val == item2:
            temp2.append(*idx)
    return np.array(temp1), np.array(temp2)


def get_markov_blankets(cg):
    mb = defaultdict(list)
    for i in range(len(cg.G.graph)):  # для каждой ноды в матрице смежности
        arr = cg.G.graph[i]
        children, parents = ind(arr, -1,
                                1)  # находим родетельские и дочерние ноды. В строке -1  - дочерняя, 1 - родительская
        # в текущей реализации достаточно просто "соседей", заранее сделал разбивку для возможности расширения
        colliders = set()
        for child_idx in children:
            temp = []
            lst = [x[child_idx] for x in cg.G.graph]  # столбец дочерней ноды в матрице смежности
            temp, _ = ind(lst, -1)  # находим родительские ноды у дочерней ноды. в столбце -1 родительская
            temp = temp[
                ~np.isin(temp, i)]  # исключаем из списка коллайдеров ноду, по которой мы пришли к данной дочерней ноде
            colliders.update(temp)
        neighbors = list(colliders | set(children) | set(parents))
        mb[i] = neighbors
    return mb


def map_names(data, mb):
    mb_mapped = defaultdict(list)
    for key, value in mb.items():
        if not list(value):
            x = data.drop(data.columns[key], axis=1)
            for i in list(x.columns):
                mb_mapped[data.columns[key]].append(i)
            print('zeros')
        else:
            for i in range(len(value)):
                mb_mapped[data.columns[key]].append(data.columns[value[i]])
    return mb_mapped


def estimate_mse(mb_mapped, data, n_splits=2):
    metric = {}
    forest = RandomForestRegressor()
    for key, value in mb_mapped.items():

        if not list(value):
            X, y = data.drop(key, axis=1), data[key]
        else:
            X, y = data[mb_mapped[key]], data[key]
        kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        kfold = kf.split(X, y)
        score = []
        for k, (train, test) in enumerate(kfold):
            forest.fit(X.iloc[train, :], y.iloc[train])
            result = mse(forest.predict(X.iloc[test, :]), y.iloc[test])
            score.append(result)
        metric[key] = np.mean(score)
    return metric, np.mean(list(metric.values()))

def estimate_mss(mb_mapped, data, n_splits=2):
    metric = {}
    model = RandomForestClassifier()

    for key, value in mb_mapped.items():
        if not list(value[0]):
            X, y = data.drop(key, axis=1), data[key]
        else:
            X, y = data[mb_mapped[key]], data[key]
        kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        kfold = kf.split(X, y)
        score = []
        for k, (train, test) in enumerate(kfold):
            model.fit(X.iloc[train, :], y.iloc[train])
            result = mcc(model.predict(X.iloc[test, :]), y.iloc[test])
            score.append(result)
        metric[key] = np.mean(score)
    return metric, np.mean(list(metric.values()))

def choose_integer_test(data, indep_test, cache_file, alpha=0.05, uc_rule=0, uc_priority=2):
    methods = {}
    launch_params = {}
    launch_results = {}
    counter = 0
    n_data = data.to_numpy()
    if indep_test is not list:
        indep_test = [indep_test]

    for j in range(len(indep_test)):
                        flag = 0
                        launch_params[counter] = 'alpha=' + str(alpha) + ' ' + 'independence test=' + str(indep_test[j])
                        methods[counter] = indep_test[j]
                        cg = pc(n_data, alpha=alpha, indep_test=indep_test[j], uc_rule=uc_rule, uc_priority=uc_priority)
                        mb = get_markov_blankets(cg)


                        mb_mapped = map_names(data, mb)
                        if indep_test == [fisherz]:
                            metrics, metrics_mean = estimate_mse(mb_mapped, data)
                        else:
                            metrics, metrics_mean = estimate_mss(mb_mapped, data)

                        launch_results[counter] = metrics_mean

                        counter += 1

    return methods, launch_results, launch_params

def get_width(data):
    sample_size = data.shape[0]
    if sample_size < 201:
        kwidthx = [0.7, 0.8, 0.9]
        kwidthy = [0.7, 0.8, 0.9]
        kwidthz = [0.35, 0.4, 0.45]
    elif sample_size < 1201:
        kwidthx = [0.4, 0.5, 0.6]
        kwidthy = [0.4, 0.5, 0.6]
        kwidthz = [0.2, 0.25, 0.3]
    else:
        kwidthx = [0.2, 0.3, 0.4]
        kwidthy = [0.2, 0.3, 0.4]
        kwidthz = [0.1, 0.15, 0.2]
    return kwidthx, kwidthy, kwidthz


def choose_cont_test(data, cache_file, indep_test=kci, alpha=0.05, uc_rule=0, uc_priority=2):
    methods = {}
    launch_params = {}
    launch_results = {}
    kernel_type = {}
    power = {}
    launch_metrics = defaultdict(dict)
    counter = 0
    n_data = data.to_numpy()
    kwidthx, kwidthy, kwidthz = get_width(n_data)
    kernel = ['Linear', 'Gaussian', 'Polynomial']
    est_width = ['empirical']
    polyd = [2, 3]
    width ={}

    #step 1 linear kernel
    launch_params[counter] = 'alpha=' + str(alpha) + ' ' + 'independence test=' + str(indep_test) + 'kernel type ' + kernel[0]
    methods[counter] = indep_test
    kernel_type[counter] = kernel[0]
    power[counter] = "None"
    cg = pc(n_data, alpha=alpha, indep_test=indep_test, kernelX = 'Linear', kernelY='Linear', uc_rule=uc_rule, uc_priority=uc_priority)
    mb = get_markov_blankets(cg)
    mb_mapped = map_names(data, mb)
    metrics, metrics_mean = estimate_mse(mb_mapped, data, 2)
    launch_results[counter] = metrics_mean
    launch_metrics[counter] = metrics
    width[counter] = None
    counter += 1

    #step 2: go thru gaussian kernel diff width

    for j in range(len(kwidthx)):
                        launch_params[counter] = 'alpha=' + str(alpha) + ' ' + 'independence test=' + str(indep_test) + 'kernel type ' + str(kernel[1]) + ' kernel width x/y/z ' + str(kwidthx[j]) + ' ' + str(kwidthy[j]) + ' ' + str(kwidthz[j])
                        methods[counter] = indep_test
                        kernel_type[counter] = kernel[1]
                        power[counter] = "None"
                        cg = pc(n_data, alpha=alpha, indep_test=indep_test, kernelX='Gaussian', kernelY='Gaussian', kwidthx=kwidthx[j], kwidthy=kwidthy[j], uc_rule=uc_rule,
                                uc_priority=uc_priority)
                        mb = get_markov_blankets(cg)
                        mb_mapped = map_names(data, mb)
                        metrics, metrics_mean = estimate_mse(mb_mapped, data, 2)
                        launch_results[counter] = metrics_mean
                        launch_metrics[counter] = metrics
                        width[counter] = kwidthx[j]
                        counter += 1
    # step 3: go thru power degrees in poly kernel

    for j in range(len(polyd)):
                        launch_params[counter] = 'alpha=' + str(alpha) + ' ' + 'independence test=' + str(indep_test) + 'kernel type ' + str(kernel[1]) + 'kernel width x/y/z ' + str(kwidthx[j]) + ' ' + str(kwidthy[j]) + ' ' + str(kwidthz[j])
                        methods[counter] = indep_test
                        kernel_type[counter] = kernel[2]
                        power[counter] = polyd[j]
                        cg = pc(n_data, alpha=alpha, indep_test=indep_test, kernelX='Polynomial', kernelY='Polynomial', polyd=polyd[j], uc_rule=uc_rule,
                                uc_priority=uc_priority)
                        mb = get_markov_blankets(cg)
                        mb_mapped = map_names(data, mb)
                        metrics, metrics_mean = estimate_mse(mb_mapped, data, 2)
                        launch_results[counter] = metrics_mean
                        launch_metrics[counter] = metrics
                        width[counter] = None
                        counter += 1

    return launch_params, launch_results, kernel_type, power, width, launch_metrics

def find_best_params(data, alpha, indep_test, date_type, n_splits=2, stable=True, uc_rule = 0, uc_priority = 0, kernel=None, polyd=None, width=None):
    methods = {}
    alphas = {}
    launch_params = {}
    launch_results = {}
    launch_metrics = defaultdict(dict)
    counter = 0
    n_data = data.to_numpy()


    if str(indep_test) != 'kci':
      for i in range(len(alpha)):
                        launch_params[counter] = 'alpha=' + str(alpha[i]) + ' ' + 'independence test=' + str(indep_test)
                        methods[counter] = indep_test
                        alphas[counter] = alpha[i]
                        cg = pc(n_data, alpha=alpha[i], indep_test=indep_test, stable=stable, uc_rule=uc_rule, uc_priority=uc_priority)
                        mb = get_markov_blankets(cg)
                        mb_mapped = map_names(data, mb)
                        print('тип данных ' + date_type)
                        if date_type == 'Дискретные':
                            metrics, metrics_mean = estimate_mse(mb_mapped, data, n_splits)
                        else:
                            metrics, metrics_mean = estimate_mss(mb_mapped, data, n_splits)
                        launch_results[counter] = metrics_mean
                        launch_metrics[counter] = metrics
                        counter += 1

    else:

        for i in range(len(alpha)):

                        flag = 0
                        launch_params[counter] = 'альфа-' + str(alpha[i]) + ' тест -' + str(indep_test) + ' тип ядра-' + str(kernel) + ' степень функции ядра-' + str(polyd) + " ширина ядра-" + str(width)
                        methods[counter] = indep_test
                        alphas[counter] = alpha[i]
                        cg = pc(n_data, alpha=alpha[i], indep_test=indep_test, stable=stable, uc_rule=uc_rule,
                        uc_priority=uc_priority)
                        mb = get_markov_blankets(cg)
                        for v in range(len(list(mb.values()))):
                            if not list(mb.values())[v]:
                                launch_results[counter] = np.inf
                                counter += 1
                                flag = 1
                                break
                        if flag == 1:
                                    continue
                        mb_mapped = map_names(data, mb)
                        metrics, metrics_mean = estimate_mse(mb_mapped, data, n_splits)
                        launch_results[counter] = metrics_mean
                        launch_metrics[counter] = metrics
                        counter += 1



    return methods, alphas, launch_results, launch_metrics, launch_params


def choose_rules(data, indep_test, alpha, date_type, uc_priority=3, uc_rule=(1,2), n_splits=2, stable=True, kernel=None, polyd=None, width=None):
    launch_results = {}
    n_data = data.to_numpy()
    for i in range(len(uc_rule)):
        if str(indep_test) != 'kci':
            cg = pc(n_data, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule[i], uc_priority=uc_priority)
        else:
            if polyd is not None:
                cg = pc(n_data, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule[i], kernel=kernel, polyd = polyd, uc_priority=uc_priority)
            elif width is not None:
                cg = pc(n_data, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule[i], kernel=kernel, width=width, uc_priority=uc_priority)
            else:
                cg = pc(n_data, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule[i], kernel=kernel, uc_priority=uc_priority)

        mb = get_markov_blankets(cg)
        mb_mapped = map_names(data, mb)
        print(date_type)
        if date_type == 'Дискретные' or date_type == 'Вещественные':
            metrics, metrics_mean = estimate_mse(mb_mapped, data, n_splits)
        else:
            metrics, metrics_mean = estimate_mss(mb_mapped, data, n_splits)
        launch_results[uc_rule[i]] = metrics_mean

    if date_type == "Дискретные":
        if launch_results[1] < launch_results[2]:
            best_rule = 1
            metric_3 = launch_results[1]
        else:
            best_rule = 2
            metric_3 = launch_results[2]
    else:
        if launch_results[1] > launch_results[2]:
            best_rule = 1
            metric_3 = launch_results[1]
        else:
            best_rule = 2
            metric_3 = launch_results[2]

    return best_rule, uc_priority, metric_3