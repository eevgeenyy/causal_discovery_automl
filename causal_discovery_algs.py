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
import crud
import schemas

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

    return methods, alphas, launch_results, \
        launch_metrics, launch_params
def get_answer(benchmark_mean, old_launch_best_alpha=None, old_launch_best_method=None, methods_dict=None,
               alphas_dict=None, launch_results=None, launch_params=None, old_best_launch=None, old_params=None, old_is_better=True):
    if launch_results: #если происходил новый запуск, считаем статистики для него
        best_result = min(launch_results, key=launch_results.get)
        best_launch = launch_results[best_result]
        best_params = launch_params[best_result]
        best_alpha = old_launch_best_alpha
        best_method = old_launch_best_method
        try: #если были запуски до этого, выбираем лучшее значение из старого и нового
            if best_launch < old_best_launch:
                best_launch = old_best_launch
                best_params = old_params
                best_alpha = alphas_dict[best_result]
                best_method = methods_dict[best_result]

                #update_bestRun(best_launch, best_params) #если значение метрики ниже - сохраняем новое лучшее значение
        except TypeError: #если запусков не было -> новый запуск становится лучшим
            old_is_better = False
    else:
        best_launch = old_best_launch
        best_params = old_params


    if best_launch < benchmark_mean:
       # cg = pc(data.to_numpy(), alpha=alphas[best_result], indep_test=methods[best_result], stable=True, uc_rule=0,
            #    uc_priority=-1)
        answer = f'Ура! Для ваших данных был подобран каузальный граф. Используемый метод - PC, параметры: {best_params}, среднее MSE - {best_launch}, MSE бенчмарка - {benchmark_mean}'
        # pyd = GraphUtils.to_pydot(cg.G, labels=data.columns)
        # pyd.write_png('graph.png')
        # await bot.send_photo(message.from_user.id, open("graph.png", 'rb'))
        # os.remove("graph.png")
    else:
        answer = f'К сожалению, для ваших данных не удалось найти подходящий причинный граф. Используемый метод - PC, параметры лучшего результата: {best_params}, MSE лучшего результата - {best_launch}, MSE бенчмарка - {benchmark_mean}'
    return answer, best_launch, best_params,best_method, best_alpha, old_is_better
