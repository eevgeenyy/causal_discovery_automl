import numpy as np
import random
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import chi2, norm
import random
#from causallearn.search.ConstraintBased.PC import pc
from itertools import permutations, combinations
from scipy.stats import chi2_contingency
from scipy.stats.contingency import crosstab
from scipy.stats import fisher_exact
from scipy.stats import power_divergence
from scipy.stats import barnard_exact
from scipy.stats import boschloo_exact
from collections import defaultdict
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()
stats = importr('stats')


def generate_ordinal_datasets(n_variables, vector_size, num_edges, scale=1, MAX=4, MIN=0):
    """
    Generate a collection of datasets with specified numbers of variables, vector sizes, data type proportions, skewness,
    and number of edges for each dataset.

    Parameters:
        n_variables (numpy.ndarray): An array of integers indicating the number of variables for each dataset.
        vector_size (numpy.ndarray): An array of integers indicating the vector size for each variable in each dataset.
        dtype_prop (numpy.ndarray): An array of floats indicating the proportion of categorical variables in each dataset.
        skew (numpy.ndarray): An array of floats indicating the skewness parameter for each variable in each dataset.
        num_edges (numpy.ndarray): An array of integers indicating the number of edges for each dataset.
        scale (float): Scale parameter for the gamma distribution.

    Returns:
        dict: A dictionary where the keys are tuples (n_vars, vec_size, dtype_prop, skew, num_edges) and the values
        are tuples representing the generated datasets and their corresponding graphs.
    """
    datasets = {}

    for v in range(len(vector_size)):
        for i in range(len(n_variables)):
            for e in range(len(num_edges)):

                # Initialize graph for current dataset
                graph = nx.DiGraph()

                # Generate a dataset with specified number of variables and vector sizes
                dataset = []
                var_count = 0  # Counter for number of variables used to generate dataset
                edges_buffer = num_edges[e]
                n_var = n_variables[i]
                while n_var:
                    if var_count > 0 and edges_buffer > 0:
                        # Generate a variable as a linear combination of some of the previous variables
                        num_combinations = np.random.randint(1, var_count + 1)
                        if edges_buffer - num_combinations < 0:
                            continue
                        combination = np.random.choice(var_count, num_combinations, replace=False)
                        weights = random.sample([1, 2, 0.5], num_combinations)

                        variable = np.round(
                            np.dot(weights, np.array(dataset)[combination, :]) + random.sample([1, -1, 0],
                                                                                               counts=[1, 1, 8],
                                                                                               k=1)).astype(int)
                        variable = np.maximum(variable, MIN)
                        variable = np.minimum(variable, MAX)
                        variable = map_array(variable)

                        graph.add_node(var_count)
                        for c in combination:
                            graph.add_edge(c, var_count)
                        edges_buffer -= num_combinations
                        var_count += 1
                        n_var -= 1

                    else:
                        # Generate a random discreet variable
                        variable = np.random.randint(1, 5, size=vector_size[v])
                        graph.add_node(var_count)
                        var_count += 1
                        n_var -= 1

                    dataset.append(variable)

                dataset = np.array(dataset)

                # Add dataset and graph to dictionary with corresponding parameters as key
                key = (n_variables[i], vector_size[v], num_edges[e])
                datasets[key] = (dataset, graph, num_edges[e] - edges_buffer)

    return datasets

def get_pValues(datasets):
  p_values_chi2 = {}
  p_values_ll = {}
  p_values_fisher = {}
  p_values_barnard = {}
  for key, dataset in datasets.items():
    data = dataset[0]
    for i in range(len(data)-2):
      for p in list(permutations([i,i+1,i+2])):
        x, y, z = p
        p_value, dof = test_conditional_independence_chi2(data, x, y, z)
        p_values_chi2[(key, p)] = (p_value, dof)
        p_value, dof = test_conditional_independence_log_likelihood(data, x, y, z)
        p_values_ll[(key, p)] = (p_value, dof)
        p_value = test_conditional_independence_fisher(data, x, y, z)
        p_values_fisher[(key, p)] = p_value
        #p_value = test_conditional_independence_barnard(data, x, y, z)
        #p_values_barnard[(key, p)] = p_value

  return p_values_chi2, p_values_ll, p_values_fisher

def one_var_disjunctio(var):
  return np.where(((var==1) | (var==3)), 1, 0)


def two_var_conjunctio(var1, var2):
  res = []
  for i in range(len(var1)):
    if var1[i] == 1 & var2[i] == 1:

      res.append(3)
    else:
      res.append(random.choices([1, 2, 3], weights=[4, 5, 1], k=1)[0])
  return res

def binom_distr(var):
  k = len(var)
  mask = (var == 1)
  var[mask] = np.array(random.choices([1, 2, 3], weights=[5, 0, 5], k=k))[mask]
  var[~mask] = 2
  return var

def nested_condition(var1, var2):
  res = []
  for a in range(len(var1)):
    if (var1[a]==1) and (var2[a]!=0):
      res.append(3)
    elif var1[a]==1:
      res.append(1)
    else:
      res.append(2)
  return res

def binary_random(var):
  k = len(var)
  mask = (var == 1)
  var[mask] = np.array(random.choices([0, 1], weights=[4, 6], k=k))[mask]
  var[~mask] = 3
  return var

def two_var_disjunctio(var1, var2):
  res = []
  for i in range(len(var1)):
    if var1[i] == 1 or var2[i] == 1:
      res.append(random.choices([1, 2, 3], weights=[1, 8, 1], k=1)[0])
    else:
      res.append(random.choices([1, 2, 3], weights=[4, 0, 4], k=1)[0])
  return res
def transform_dict(p_values):
  p_v = {}
  for key, value in p_values.items():
    first_tuple, second_tuple = key
    if first_tuple not in p_v:
        p_v[first_tuple] = []
    p_v[first_tuple].append({second_tuple: value})
  return p_v


alpha = 0.05


# Function to infer independence between nodes given p-value
def is_independent(p_value):
    return p_value > alpha


# Function to calculate true negatives, true positives, false negatives, false positives,
# precision and recall for a given graph and its associated p-values
def calculate_metrics(graph, p_values):
    tp, tn, fp, fn = 0, 0, 0, 0

    for combination in p_values:
        p_value = list(combination.values())[0][0]
        x, y, z = list(combination.keys())[0]

        if is_independent(p_value):
            if not nx.d_separated(graph, {x}, {y}, {z}):
                fp += 1
            else:
                tn += 1
        else:
            if nx.d_separated(graph, {x}, {y}, {z}):
                fn += 1
            else:
                tp += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return tn, tp, fn, fp, precision, recall

def map_array(arr):
    unique_vals = np.unique(arr)
    cardinality = len(unique_vals)
    mapping_dict = {val: i for i, val in enumerate(unique_vals)}
    mapped_arr = np.array([mapping_dict[val] for val in arr])
    return mapped_arr

one_var_rules = [one_var_disjunctio, binom_distr, binary_random]
two_var_rules = [two_var_conjunctio, nested_condition, two_var_disjunctio]
def generate_categorical_datasets_from_1_variable(n_variables, vector_size, num_edges, one_var_rules= one_var_rules):
    """
    Generate a collection of datasets with specified numbers of variables, vector sizes and number of edges for each dataset.

    Parameters:
        n_variables (numpy.ndarray): An array of integers indicating the number of variables for each dataset.
        vector_size (numpy.ndarray): An array of integers indicating the vector size for each variable in each dataset.
        num_edges (numpy.ndarray): An array of integers indicating the number of edges for each dataset.

    Returns:
        dict: A dictionary where the keys are tuples (n_vars, vec_size, rule_nuber, num_edges) and the values
        are tuples representing the generated datasets and their corresponding graphs.
    """
    datasets = {}

    rule_counter = 0
    for v in range(len(vector_size)):
        for i in range(len(n_variables)):
            for e in range(len(num_edges)):

                # Initialize graph for current dataset
                graph = nx.DiGraph()

                # Generate a dataset with specified number of variables and vector sizes
                dataset = []
                var_count = 0  # Counter for number of variables used to generate dataset
                edges_buffer = num_edges[e]
                n_var = n_variables[i]

                while n_var:
                    if rule_counter >= 3:
                        rule_counter = 0
                    if var_count > 0 and edges_buffer > 0:
                        # Generate a variable as a linear combination of some of the previous variables

                        #  num_combinations = np.random.randint(1, var_count+1)
                        #  if edges_buffer - num_combinations < 0:
                        #     continue
                        parent_variable_n = np.random.choice(var_count, 1, replace=False)[0]
                        variable = one_var_rules[rule_counter](np.array(dataset)[parent_variable_n, :])

                        graph.add_node(var_count + 1)

                        graph.add_edge(parent_variable_n + 1, var_count + 1)
                        edges_buffer -= 1
                        var_count += 1
                        n_var -= 1
                        rule_counter += 1

                    else:
                        # Generate a categorical variable
                        variable = np.random.randint(0, 3, size=vector_size[v])
                        graph.add_node(var_count + 1)
                        var_count += 1
                        n_var -= 1

                    dataset.append(variable)

                dataset = np.array(dataset)

                # Add dataset and graph to dictionary with corresponding parameters as key
                key = (n_variables[i], vector_size[v], num_edges[e])
                datasets[key] = (dataset, graph, num_edges[e] - edges_buffer)

    return datasets


def generate_categorical_datasets_from_2_variables(n_variables, vector_size, num_edges, two_var_rules=two_var_rules):
    """
    Generate a collection of datasets with specified numbers of variables, vector sizes and number of edges for each dataset.

    Parameters:
        n_variables (numpy.ndarray): An array of integers indicating the number of variables for each dataset.
        vector_size (numpy.ndarray): An array of integers indicating the vector size for each variable in each dataset.
        num_edges (numpy.ndarray): An array of integers indicating the number of edges for each dataset.

    Returns:
        dict: A dictionary where the keys are tuples (n_vars, vec_size, rule_nuber, num_edges) and the values
        are tuples representing the generated datasets and their corresponding graphs.
    """
    datasets = {}

    rule_counter = 0
    for v in range(len(vector_size)):
        for i in range(len(n_variables)):
            for e in range(len(num_edges)):

                # Initialize graph for current dataset
                graph = nx.DiGraph()

                # Generate a dataset with specified number of variables and vector sizes
                dataset = []
                var_count = 0  # Counter for number of variables used to generate dataset
                edges_buffer = num_edges[e]
                n_var = n_variables[i]

                while n_var:
                    if rule_counter >= 3:
                        rule_counter = 0
                    if var_count > 1 and edges_buffer > 0:
                        # Generate a variable as a linear combination of some of the previous variables

                        parent_variables_n = np.random.choice(var_count, 2, replace=False)

                        variable = two_var_rules[rule_counter](np.array(dataset)[parent_variables_n[0], :],
                                                               np.array(dataset)[parent_variables_n[1], :])

                        graph.add_node(var_count + 1)

                        graph.add_edge(parent_variables_n[0] + 1, var_count + 1)
                        graph.add_edge(parent_variables_n[1] + 1, var_count + 1)
                        edges_buffer -= 1
                        var_count += 1
                        n_var -= 1
                        rule_counter += 1

                    else:
                        # Generate a categorical variable
                        variable = np.random.randint(0, 3, size=vector_size[v])
                        graph.add_node(var_count + 1)
                        var_count += 1
                        n_var -= 1

                    dataset.append(variable)

                dataset = np.array(dataset)

                # Add dataset and graph to dictionary with corresponding parameters as key
                key = (n_variables[i], vector_size[v], num_edges[e])
                datasets[key] = (dataset, graph, num_edges[e] - edges_buffer)

    return datasets

