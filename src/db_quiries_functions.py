from causallearn.utils.cit import fisherz
# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import gsq
from causallearn.utils.cit import mv_fisherz, chisq
from causallearn.utils.cit import kci
from src import crud
from src import models


indep_test_discreet = [fisherz]
indep_test_cont = [kci]
indep_test_cat = [gsq]

alpha_cons = [0.05, 0.1, 0.15]
alpha_aggr = [0.2, 0.25, 0.3]
alpha_nopref = [0.05, 0.1]

preferences = ['Лучше, если будут отсутствовать некоторые существующие связи',
                            'Лучше, если будет присутствовать связи, которых на самом деле нет',
                            'Нет предпочтений']
data_types = ['Дискретные', 'Категориальные', 'Вещественные']

def get_db():
    db = models.SessionLocal()
    try:
        yield db
    finally:
        db.close()
def check_history(url, data_type, preference):
    new_methods =None
    if preference == preferences[0]:
        alpha = alpha_cons
    elif preference == preferences[1]:
        alpha = alpha_aggr
    else:
        alpha = alpha_nopref

    if data_type == data_types[0]:
        methods = indep_test_discreet
    elif data_type == data_types[1]:
        methods = indep_test_cat
    else:
        methods = indep_test_cont

    try:
        dataset = crud.get_dataset(url, models.SessionLocal())
        best_runid = dataset.best_run
        best_run = crud.get_best_run(best_runid, models.SessionLocal())
        if best_run.best_alpha == 0:
            best_alpha = 0.05
        else:
            best_alpha = best_run.best_alpha

        prev_launch_info = tuple([best_run.methods, best_run.alphas, best_run.best_result, best_run.best_test, best_alpha, best_run.uc_rule, best_run.uc_priority, dataset.baseline])

        new_alphas = list(set(alpha) - set(prev_launch_info[1]))

        if not new_methods and not new_alphas:
            status, new_params = 0, None
        else:
            status = 1
            new_params = (new_methods, new_alphas)
    except Exception:

            status, prev_launch_info, new_params = 2, None, None

    # проверяем, появились ли с момента последнего запуска какие-либо изменения в наборе гиперпараметров.
    # если появились, запустим отдельно для них поиск лучших и сравним с предыдущим лучшим результатом

    return new_params, status, prev_launch_info
