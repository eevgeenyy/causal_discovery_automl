from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import gsq
from causallearn.utils.cit import mv_fisherz
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse


# “fisherz”: Fisher’s Z conditional independence test.

# “chisq”: Chi-squared conditional independence test.

# “gsq”: G-squared conditional independence test.

# “mv_fisherz”: Missing-value Fisher’s Z conditional independence test.

def get_benchmark(df, n_splits):
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
        for i in range(len(value)):
            mb_mapped[data.columns[key]].append(data.columns[value[i]])
    return mb_mapped


def estimate(mb_mapped, data, n_splits):
    metric = {}
    forest = RandomForestRegressor()

    for key, value in mb_mapped.items():
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


def find_best_params(data, alpha, indep_test, n_splits, stable=True, uc_rule=0, uc_priority=-1):
    benchmark, benchmark_mean = get_benchmark(data, n_splits)
    methods = {}
    alphas = {}
    launch_params = {}
    launch_results = {}
    launch_metrics = defaultdict(dict)
    counter = 0
    n_data = data.to_numpy()
    for i in range(len(alpha)):
        for j in range(len(indep_test)):
            flag = 0
            launch_params[counter] = 'alpha=' + str(alpha[i]) + ' ' + 'independence test=' + str(indep_test[j])
            methods[counter] = indep_test[j]
            alphas[counter] = alpha[i]
            cg = pc(n_data, alpha=alpha[i], indep_test=indep_test[j], stable=stable, uc_rule=uc_rule,
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
            metrics, metrics_mean = estimate(mb_mapped, data, n_splits)
            launch_results[counter] = metrics_mean
            launch_metrics[counter] = metrics
            counter += 1

    return benchmark, benchmark_mean, methods, alphas, launch_results, \
        launch_metrics, launch_params