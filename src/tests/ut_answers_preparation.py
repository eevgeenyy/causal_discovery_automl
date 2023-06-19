from src.answers_preparation import *

def test_step2_answer():
    # Test case 1: old_launch_best_alpha != best_alpha, metric_type == 'MSS', metric_2 > metric_1
    best_alpha = 0.5
    old_launch_best_alpha = 0.3
    metric_type = 'MSS'
    metric_2 = 0.8
    metric_1 = 0.6
    expected_answ = "удалось улучшить значение коэффициента Мэтью на 0.2 пункта, лучшее значение alpha: 0.5"
    expected_best_alpha = 0.5
    assert step2_answer(best_alpha, old_launch_best_alpha, metric_type, metric_2, metric_1) == (expected_answ, expected_best_alpha)

    # Test case 2: old_launch_best_alpha != best_alpha, metric_type == 'MSS', metric_2 <= metric_1
    best_alpha = 0.5
    old_launch_best_alpha = 0.3
    metric_type = 'MSS'
    metric_2 = 0.6
    metric_1 = 0.8
    expected_answ = "значение метрики на этом шаге не улучшилось, лучшее значение alpha по прежнему: 0.3"
    expected_best_alpha = 0.3
    assert step2_answer(best_alpha, old_launch_best_alpha, metric_type, metric_2, metric_1) == (expected_answ, expected_best_alpha)

    # Test case 3: old_launch_best_alpha != best_alpha, metric_type != 'MSS', metric_1 > metric_2
    best_alpha = 0.5
    old_launch_best_alpha = 0.5
    metric_type = 'other'
    metric_2 = 0.3
    metric_1 = 0.4
    expected_answ = "удалось улучшить значение MSE на 0.1, лучшее значение alpha: 0.5"
    expected_best_alpha = 0.5
    assert step2_answer(best_alpha, old_launch_best_alpha, metric_type, metric_2, metric_1) == (expected_answ, expected_best_alpha)

    # Test case 4: old_launch_best_alpha != best_alpha, metric_type != 'MSS', metric_1 <= metric_2
    best_alpha = 0.5
    old_launch_best_alpha = 0.5
    metric_type = 'other'
    metric_2 = 0.4
    metric_1 = 0.3
    expected_answ = "Лучшее значение alpha оказалось тем же, что и в предыдущем запуске: 0.5"
    expected_best_alpha = 0.5
    assert step2_answer(best_alpha, old_launch_best_alpha, metric_type, metric_2, metric_1) == (expected_answ, expected_best_alpha)

    # Test case 5: old_launch_best_alpha == best_alpha
    best_alpha = 0.5
    old_launch_best_alpha = 0.5
    metric_type = 'MSS'
    metric_2 = 0.8
    metric_1 = 0.6
    expected_answ = "Лучшее значение alpha оказалось тем же, что и в предыдущем запуске"
    expected_best_alpha = 0.5
    assert step2_answer(best_alpha, old_launch_best_alpha, metric_type, metric_2, metric_1) == (
    expected_answ, expected_best_alpha)

def test_step3_answer():
    # Test case 1: metric_type == 'MSS', metric_3 > metric_2
    best_uc_rule = 1
    best_uc_priority = 0
    metric_type = 'MSS'
    metric_3 = 0.8
    metric_2 = 0.6
    expected_answ = "удалось улучшить значение коэффициента Мэтью на 0.2 пункта, лучшее значения: ориентировать неопределенные триплеты X-Y-Z с помощью дополнительного CI теста, приоретизация более сильных коллайдеров"
    expected_best_metric = 0.8
    assert step3_answer(best_uc_rule, best_uc_priority, metric_type, metric_3, metric_2) == (expected_answ, expected_best_metric)

    # Test case 2: metric_type == 'MSS', metric_3 <= metric_2
    best_uc_rule = 2
    best_uc_priority = 0
    metric_type = 'MSS'
    metric_3 = 0.6
    metric_2 = 0.8
    expected_answ = "значение метрики на этом шаге не улучшилось, лучшее значения по прежнему: ориентировать разделяющее множество, приоретизация более сильных коллайдеров"
    expected_best_metric = 0.8
    assert step3_answer(best_uc_rule, best_uc_priority, metric_type, metric_3, metric_2) == (expected_answ, expected_best_metric)

    # Test case 3: metric_type != 'MSS', metric_2 > metric_3
    best_uc_rule = 0
    best_uc_priority = 1
    metric_type = 'other'
    metric_3 = 0.4
    metric_2 = 0.6
    expected_answ = "удалось улучшить значение MSE на 0.2 пункта, лучшее значения правила ориентации коллайдеров: использовать разделяющее множество, приоретизация более сильных коллайдеров"
    expected_best_metric = 0.4
    assert step3_answer(best_uc_rule, best_uc_priority, metric_type, metric_3, metric_2) == (expected_answ, expected_best_metric)

    # Test case 4: metric_type != 'MSS', metric_2 <= metric_3
    best_uc_rule = 1
    best_uc_priority = 1
    metric_type = 'other'
    metric_3 = 0.6
    metric_2 = 0.4
    expected_answ = "значение метрики на этом шаге не улучшилось, лучшее значения по прежнему: ориентировать неопределенные триплеты X-Y-Z с помощью дополнительного CI теста, приоретизация более сильных коллайдеров"
    expected_best_metric = 0.6
    assert step3_answer(best_uc_rule, best_uc_priority, metric_type, metric_3, metric_2) == (expected_answ, expected_best_metric)

def test_get_answer():
    # Test case 1: New launch results are better than the old best launch
    benchmark_mean = 0.5
    old_launch_best_alpha = 0.3
    old_launch_best_method = 'Method A'
    methods_dict = {'Method A': 1, 'Method B': 2}
    alphas_dict = {'Alpha 1': 0.1, 'Alpha 2': 0.2}
    launch_results = {'Result 1': 0.4, 'Result 2': 0.3}
    launch_params = {'Result 1': {'param1': 1, 'param2': 2}, 'Result 2': {'param1': 3, 'param2': 4}}
    old_best_launch = 0.6
    old_params = {'param1': 5, 'param2': 6}
    old_is_better = True
    expected_answer = "Ура! Для ваших данных был подобран каузальный граф. Используемый метод - PC, параметры: {'param1': 3, 'param2': 4}, среднее MSE - 0.3, MSE бенчмарка - 0.5"
    expected_best_launch = 0.3
    expected_best_params = {'param1': 3, 'param2': 4}
    expected_best_method = 'Method A'
    expected_best_alpha = 0.3
    expected_old_is_better = False
    assert get_answer(benchmark_mean, old_launch_best_alpha, old_launch_best_method, methods_dict, alphas_dict, launch_results, launch_params, old_best_launch, old_params, old_is_better) == (expected_answer, expected_best_launch, expected_best_params, expected_best_method, expected_best_alpha, expected_old_is_better)

    # Test case 2: New launch results are worse than the old best launch
    benchmark_mean = 0.5
    old_launch_best_alpha = 0.3
    old_launch_best_method = 'Method A'
    methods_dict = {'Method A': 1, 'Method B': 2}
    alphas_dict = {'Alpha 1': 0.1, 'Alpha 2': 0.2}
    launch_results = {'Result 1': 0.6, 'Result 2': 0.7}
    launch_params = {'Result 1': {'param1': 1, 'param2': 2}, 'Result 2': {'param1': 3, 'param2': 4}}
    old_best_launch = 0.4
    old_params = {'param1': 5, 'param2': 6}
    old_is_better = False
    expected_answer = "К сожалению, для ваших данных не удалось найти подходящий причинный граф. Используемый метод - PC, параметры лучшего результата: {'param1': 5, 'param2': 6}, MSE лучшего результата - 0.4, MSE бенчмарка - 0.5"
    expected_best_launch = 0.4
    expected_best_params = {'param1': 5, 'param2': 6}
    expected_best_method = 'Method A'
    expected_best_alpha = 0.3
    expected_old_is_better = False
    assert get_answer(benchmark_mean, old_launch_best_alpha, old_launch_best_method, methods_dict, alphas_dict, launch_results, launch_params, old_best_launch, old_params, old_is_better) == (expected_answer, expected_best_launch, expected_best_params, expected_best_method, expected_best_alpha, expected_old_is_better)

    # Test case 3: No new launch results
    benchmark_mean = 0.5
    old_launch_best_alpha = 0.3
    old_launch_best_method = 'Method A'
    methods_dict = {'Method A': 1, 'Method B': 2}
    alphas_dict = {'Alpha 1': 0.1, 'Alpha 2': 0.2}
    launch_results = None
    launch_params = None
    old_best_launch = 0.4
    old_params = {'param1': 5, 'param2': 6}
    old_is_better = False
    expected_answer = "К сожалению, для ваших данных не удалось найти подходящий причинный граф. Используемый метод - PC, параметры лучшего результата: {'param1': 5, 'param2': 6}, MSE лучшего результата - 0.4, MSE бенчмарка - 0.5"
    expected_best_launch = 0.4
    expected_best_params = {'param1': 5, 'param2': 6}
    expected_best_method = 'Method A'
    expected_best_alpha = 0.3
    expected_old_is_better = False
    assert get_answer(benchmark_mean, old_launch_best_alpha, old_launch_best_method, methods_dict, alphas_dict, launch_results, launch_params, old_best_launch, old_params, old_is_better) == (expected_answer, expected_best_launch, expected_best_params, expected_best_method, expected_best_alpha, expected_old_is_better)


test_step2_answer()
test_step3_answer()
test_get_answer()

