

def step2_answer(best_alpha, old_launch_best_alpha, metric_type, metric_2, metric_1):
    if old_launch_best_alpha != best_alpha:
            if metric_type == 'MSS':
                if metric_2 > metric_1:
                        answ = f'удалось улучшить значение коэффициента Мэтью на {metric_2 - metric_1} пункта, лучшее значение alpha: {best_alpha}'
                else:
                   answ = f'значение метрики на этом шаге не улучшилось, лучшее значение alpha по прежнему: {old_launch_best_alpha}'
                   best_alpha = old_launch_best_alpha
            else:
                if metric_1 > metric_2:
                    answ = f'удалось улучшить значение MSE на {metric_1 - metric_2}, лучшее значение alpha: {best_alpha}'
                else:
                    answ = f'Лучшее значение alpha оказалось тем же, что и в предыдущем запуске: {old_launch_best_alpha}'
    else:
        answ = f'Лучшее значение alpha оказалось тем же, что и в предыдущем запуске: {old_launch_best_alpha}'
        best_alpha = old_launch_best_alpha

    return answ, best_alpha

def step3_answer(best_uc_rule, best_uc_priority, metric_type, metric_3, metric_2):
    default_uc_rule = 0
    default_uc_priority = 0
    rules_dict = {0: 'использовать разделяющее множество', 1: "ориентировать неопределенные триплеты X-Y-Z с помощью дополительного CI теста", 2: 'ориентировать только определенные коллайдеры'}
    if metric_type == 'MSS':
                if metric_3 > metric_2:
                        answ = f'удалось улучшить значение коэффициента Мэтью на {metric_3 - metric_2} пункта, лучшее значения: {rules_dict[best_uc_rule]}, приоретизация более сильных коллайдеров'
                        best_metric = metric_3
                else:
                   answ = f'значение метрики на этом шаге не улучшилось, лучшее значения по прежнему: {rules_dict[default_uc_rule]}, приоретизация более сильных коллайдеров'
                   best_metric = metric_2
    else:
                if metric_2 > metric_3:
                    answ = f'удалось улучшить значение MSE на {metric_2 - metric_3} пункта, лучшее значения правила ориентации коллайдеров: {rules_dict[best_uc_rule]}, приоретизация более сильных коллайдеров'
                    best_metric = metric_3
                else:
                   answ = f'значение метрики на этом шаге не улучшилось, лучшее значения по прежнему: {rules_dict[default_uc_rule]}, приоретизация более сильных коллайдеров'
                   best_metric = metric_2
    return answ, best_metric

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
                best_alpha = old_launch_best_alpha
                best_method = old_launch_best_method

                #update_bestRun(best_launch, best_params) #если значение метрики ниже - сохраняем новое лучшее значение
        except TypeError: #если запусков не было -> новый запуск становится лучшим
            old_is_better = False
    else:
        best_launch = old_best_launch
        best_params = old_params
        best_alpha = old_launch_best_alpha
        best_method = old_launch_best_method


    if best_launch < benchmark_mean:

        answer = f'Ура! Для ваших данных был подобран каузальный граф. Используемый метод - PC, параметры: {best_params}, среднее MSE - {best_launch}, MSE бенчмарка - {benchmark_mean}'

    else:
        answer = f'К сожалению, для ваших данных не удалось найти подходящий причинный граф. Используемый метод - PC, параметры лучшего результата: {best_params}, MSE лучшего результата - {best_launch}, MSE бенчмарка - {benchmark_mean}'
    return answer, best_launch, best_params,best_method, best_alpha, old_is_better