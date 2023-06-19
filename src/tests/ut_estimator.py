from src.estimator import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef as mcc
import numpy as np
from collections import defaultdict
def test_get_benchmark_cat():
    # Test case 1: Empty DataFrame
    df = pd.DataFrame()
    expected_benchmark = {}
    expected_mean_score = 0.0
    assert get_benchmark_cat(df) == (expected_benchmark, expected_mean_score)

    # Test case 2: Non-empty DataFrame
    df = pd.DataFrame({'A': [1, 0, 1, 0, 1], 'B': [0, 0, 1, 1, 1]})
    expected_benchmark = {'A': 0.2, 'B': 0.4}
    expected_mean_score = 0.3
    assert get_benchmark_cat(df) == (expected_benchmark, expected_mean_score)

    # Test case 3: DataFrame with missing values
    df = pd.DataFrame({'A': [1, 0, 1, None, 1], 'B': [0, 0, 1, 1, None]})
    expected_benchmark = {'A': 0.2, 'B': 0.4}
    expected_mean_score = 0.3
    assert get_benchmark_cat(df) == (expected_benchmark, expected_mean_score)

def test_get_benchmark_cont():
    # Test case 1: Empty DataFrame
    df = pd.DataFrame()
    expected_benchmark = {}
    expected_mean_score = 0.0
    assert get_benchmark_cont(df) == (expected_benchmark, expected_mean_score)

    # Test case 2: Non-empty DataFrame
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})
    expected_benchmark = {'A': 1.25, 'B': 5.0}
    expected_mean_score = 3.125
    assert get_benchmark_cont(df) == (expected_benchmark, expected_mean_score)

    # Test case 3: DataFrame with missing values
    df = pd.DataFrame({'A': [1, 2, 3, None, 5], 'B': [2, 4, 6, 8, None]})
    expected_benchmark = {'A': 1.25, 'B': 5.0}
    expected_mean_score = 3.125
    assert get_benchmark_cont(df) == (expected_benchmark, expected_mean_score)

def test_ind():
    # Test case 1: Single occurrence of item1
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    item1 = 5
    expected_temp1 = np.array([(1, 1)])
    expected_temp2 = np.array([])
    assert np.array_equal(ind(array, item1), (expected_temp1, expected_temp2))

    # Test case 2: Multiple occurrences of item1 and item2
    array = np.array([[1, 2, 2], [4, 5, 6], [7, 8, 9]])
    item1 = 2
    item2 = 6
    expected_temp1 = np.array([(0, 1), (0, 2)])
    expected_temp2 = np.array([(1, 2)])
    assert np.array_equal(ind(array, item1, item2), (expected_temp1, expected_temp2))

    # Test case 3: No occurrences of item1 and item2
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    item1 = 0
    item2 = 10
    expected_temp1 = np.array([])
    expected_temp2 = np.array([])
    assert np.array_equal(ind(array, item1, item2), (expected_temp1, expected_temp2))

def test_get_markov_blankets():
    # Test case 1: Simple directed graph with one parent and one child
    cg = np.array([[0, -1, 0], [0, 0, 1], [0, 0, 0]])
    expected_mb = defaultdict(list, {0: [1], 1: [0, 2], 2: [1]})
    assert dict(get_markov_blankets(cg)) == dict(expected_mb)

    # Test case 2: Directed graph with multiple parents and children
    cg = np.array([[0, -1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    expected_mb = defaultdict(list, {0: [1], 1: [0, 2], 2: [1], 3: []})
    assert dict(get_markov_blankets(cg)) == dict(expected_mb)

    # Test case 3: Empty graph
    cg = np.array([])
    expected_mb = defaultdict(list)
    assert dict(get_markov_blankets(cg)) == dict(expected_mb)

def test_map_names():
    # Test case 1: Mapping names with empty Markov blankets
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    mb = defaultdict(list, {0: [], 1: [], 2: []})
    expected_mb_mapped = defaultdict(list, {'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']})
    assert dict(map_names(data, mb)) == dict(expected_mb_mapped)

    # Test case 2: Mapping names with non-empty Markov blankets
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    mb = defaultdict(list, {0: [1], 1: [0, 2], 2: [1]})
    expected_mb_mapped = defaultdict(list, {'A': ['B'], 'B': ['A', 'C'], 'C': ['B']})
    assert dict(map_names(data, mb)) == dict(expected_mb_mapped)

    # Test case 3: Empty data and Markov blankets
    data = pd.DataFrame()
    mb = defaultdict(list)
    expected_mb_mapped = defaultdict(list)
    assert dict(map_names(data, mb)) == dict(expected_mb_mapped)

def test_estimate_mse():
    # Test case 1: Estimate MSE with mapped Markov blankets and non-empty data
    mb_mapped = defaultdict(list, {'A': ['B'], 'B': ['A', 'C'], 'C': ['B']})
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    expected_metric = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    expected_mean_metric = 0.0
    metric, mean_metric = estimate_mse(mb_mapped, data)
    assert metric == expected_metric
    assert mean_metric == expected_mean_metric

    # Test case 2: Estimate MSE with mapped Markov blankets and empty data
    mb_mapped = defaultdict(list, {'A': ['B'], 'B': ['A', 'C'], 'C': ['B']})
    data = pd.DataFrame()
    expected_metric = {}
    metric, mean_metric = estimate_mse(mb_mapped, data)
    assert metric == expected_metric
    assert np.isnan(mean_metric)

    # Test case 3: Estimate MSE with empty mapped Markov blankets and non-empty data
    mb_mapped = defaultdict(list, {'A': [], 'B': [], 'C': []})
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    expected_metric = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    expected_mean_metric = 0.0
    metric, mean_metric = estimate_mse(mb_mapped, data)
    assert metric == expected_metric
    assert mean_metric == expected_mean_metric

    # Test case 4: Estimate MSE with empty mapped Markov blankets and empty data
    mb_mapped = defaultdict(list, {'A': [], 'B': [], 'C': []})
    data = pd.DataFrame()
    expected_metric = {}
    metric, mean_metric = estimate_mse(mb_mapped, data)
    assert metric == expected_metric
    assert np.isnan(mean_metric)
import numpy as np

def test_choose_integer_test():
    # Test case 1: Choose integer test with one independence test
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indep_test = lambda x: x  # A dummy independence test function
    alpha = 0.05
    uc_rule = 0
    uc_priority = 2
    expected_methods = {0: indep_test}
    expected_launch_params = {0: f"alpha={alpha} independence test=<function test_choose_integer_test.<locals>.<lambda> at 0x0000000000000000>"}
    expected_launch_results = {0: 0.0}
    methods, launch_results, launch_params = choose_integer_test(data, indep_test, alpha, uc_rule, uc_priority)
    assert methods == expected_methods
    assert launch_params == expected_launch_params
    assert launch_results == expected_launch_results

    # Test case 2: Choose integer test with multiple independence tests
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indep_test = [lambda x: x, lambda x: x ** 2]  # Dummy independence test functions
    alpha = 0.05
    uc_rule = 0
    uc_priority = 2
    expected_methods = {0: indep_test[0], 1: indep_test[1]}
    expected_launch_params = {
        0: f"alpha={alpha} independence test=<function test_choose_integer_test.<locals>.<lambda> at 0x0000000000000000>",
        1: f"alpha={alpha} independence test=<function test_choose_integer_test.<locals>.<lambda> at 0x0000000000000000>"
    }
    expected_launch_results = {0: 0.0, 1: 0.0}
    methods, launch_results, launch_params = choose_integer_test(data, indep_test, alpha, uc_rule, uc_priority)
    assert methods == expected_methods
    assert launch_params == expected_launch_params
    assert launch_results == expected_launch_results




test_choose_integer_test()
test_estimate_mse()
test_map_names()
test_get_markov_blankets()
test_ind()
test_get_benchmark_cont()
test_get_benchmark_cat()
