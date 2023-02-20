import pika
from fastapi import FastAPI, Depends
import models
import crud
import schemas
from causal_discovery_algs import *
from sqlalchemy.orm import Session


indep_test = [gsq, mv_fisherz, fisherz]
alpha = [0.05, 0.2, 0.8]
n_splits = 2
out = "."
app = FastAPI()

def get_db():
    db = models.SessionLocal()
    try:
        yield db
    finally:
        db.close()

"""
@app.post('/user', response_model=schemas.RunBase)
async def get_dataset(url, db: Session = Depends(get_db)):
    return crud.get_dataset(db, url)

@app.post('/user', response_model=schemas.RunBase)
async def get_stored_results(best_runid, dataset: schemas.RunBase, db: Session = Depends(get_db)):
    return crud.get_best_run(best_runid, db)
@app.post('/user', response_model=schemas.DatasetBase)
async def create_dataset(dataset, scheme: schemas.DatasetBase, db: Session = Depends(get_db)):
    return crud.create_dataset(dataset, db, scheme)

@app.post('/user', response_model=schemas.RunBase)
async def create_run(dataset: schemas.RunBase, db: Session = Depends(get_db)):
    return crud.create_run(db)

@app.post('/user', response_model=schemas.DatasetBase)
async def update_best_run(run_id, dataset_id, dataset: schemas.DatasetBase, db: Session = Depends(get_db)):
    pass
"""
def get_result(url, old_is_better=False, new_results=False):
    # проверяем, делали ли мы расчет для датасета, сохраненного по данной ссылке
    data = pd.read_csv(url)
    try:
        dataset = crud.get_dataset(models.SessionLocal(), url)
        best_runid = dataset.best_run
        benchmark_mean = dataset.baseline
        best_run = crud.get_best_run(models.SessionLocal(), best_runid)
        print('got dataset')
        old_methods, old_alphas, old_launch_results, old_launch_best_method, old_launch_best_alpha, old_launch_params = best_run.methods, \
        best_run.alphas, best_run.best_result, best_run.best_method, best_run.best_alpha, best_run.best_params

    # проверяем, появились ли с момента последнего запуска какие-либо изменения в наборе гиперпараметров.
    # если появились, запустим отдельно для них поиск лучших и сравним с предыдущим лучшим результатом
        new_methods = set(indep_test) - set(old_methods)
        new_alphas = set(alpha) - set(old_alphas)

        if new_methods:
        # проверяем новые методы по всем альфа
           new_results = True
           methods_dict, alphas_dict, new_launch_results, new_launch_metrics, new_launch_params = find_best_params(data,
                                                                                                                alpha,
                                                                                                                new_methods,
                                                                                                                n_splits)
        # ответ зависит от сравнения прошлого лучшего реультата и результата новых методов
           answer, best_launch, best_params, best_method, best_alpha, old_is_better = get_answer(launch_results=new_launch_results,
                                                                               launch_params=new_launch_params,
                                                                               old_best_launch=old_launch_results,
                                                                               old_params=old_launch_params,
                                                                               benchmark_mean=benchmark_mean,
                                                                               methods_dict=methods_dict,
                                                                               alphas_dict=alphas_dict,
                                                                               old_launch_best_method=old_launch_best_method,
                                                                               old_launch_best_alpha=old_launch_best_alpha)
           run = models.Runs(
           dataset_id=dataset.id,
           best_result=best_launch,
           best_algorithm='PC',
           best_method=best_method,
           best_alpha=best_alpha,
           alphas=alpha,
           methods=new_methods)
           print('new methods')

           crud.create_run(run, models.SessionLocal())
           print('run created')

        elif new_alphas:
        # проверяем старые методы по новым альфам
           new_results = True
           methods_dict, alphas_dict, new_launch_results, new_launch_metrics, new_launch_params = find_best_params(data,
                                                                                                                new_alphas,
                                                                                                                indep_test,
                                                                                                                n_splits)
        # ответ зависит от сравнения прошлого лучшего реультата и результата по новым альфа
           answer, best_launch, best_params, best_method, best_alpha, old_is_better = get_answer(
            launch_results=new_launch_results,
            launch_params=new_launch_params,
            old_best_launch=old_launch_results,
            old_params=old_launch_params,
            benchmark_mean=benchmark_mean,
            methods_dict=methods_dict,
            alphas_dict=alphas_dict,
            old_launch_best_method=old_launch_best_method,
            old_launch_best_alpha=old_launch_best_alpha)
           print('new alphas')
           run = models.Runs(
            dataset_id=dataset.id,
            best_result=best_launch,
            best_algorithm='PC',
            best_method=best_method,
            best_alpha=best_alpha,
            alphas=new_alphas,
            methods=indep_test)
           crud.create_run(run, models.SessionLocal())
           print('run created')


        else:
        # если новых гиперпараметров нет, то формируем ответ по старым результатам
            print('no new results')
            answer, best_launch, best_params, best_method, best_alpha, old_is_better = get_answer(old_best_launch=old_launch_results,
                                                                               old_params=old_launch_params,
                                                                               benchmark_mean=benchmark_mean,
                                                                               old_launch_best_method=old_launch_best_method,
                                                                               old_launch_best_alpha=old_launch_best_alpha)
            print('got answer')

    except Exception as e:
# если не удалось найти предыдущие запуски, считаем заново
       new_results = True
       print('no records')
       benchmark, benchmark_mean = get_benchmark(data, n_splits)
       methods_dict, alphas_dict, new_launch_results, new_launch_metrics, new_launch_params = find_best_params(
       data, alpha, indep_test, n_splits)
       answer, best_launch, best_params, best_method, best_alpha, old_is_better = get_answer(launch_results=new_launch_results,
                                                                       launch_params=new_launch_params,
                                                                       benchmark_mean=benchmark_mean,
                                                                       methods_dict=methods_dict,
                                                                       alphas_dict=alphas_dict)
       print('got answer')

       dataset = models.Datasets(
                 dataset_link=url,
                 rows_number=data.shape[0],
                 columns_number=data.shape[1],
                 baseline=benchmark_mean)

       crud.create_dataset(dataset, models.SessionLocal())
       print('dataset created')
       dataset = crud.get_dataset(models.SessionLocal(), url)
       print('got dataset')

       run = models.Runs(
             dataset_id=dataset.id,
             best_result=best_launch,
             best_algorithm='PC',
             best_method=best_method,
             best_alpha=best_method,
             alphas=alpha,
             methods=indep_test)
       crud.create_run(run, models.SessionLocal())
       print('run created')
    #if not old_is_better:
      # update_best_run(dataset_id = dataset.id, best_runid= new_run.id)
    return answer

def record_results():
    pass
def update_best_rus():
    pass

def on_request_message_received(ch, method, properties, body):
    print(f"Received Request: {properties.correlation_id}")
    #print(body)
    #"answer = get_result(body.decode('utf8'))
    #
    with open(f'{properties.correlation_id}.txt', 'w') as f:
        f.write(properties.correlation_id)

    with open('ready.txt', 'w') as f:
        f.write('1')
    #ch.basic_publish('', routing_key=properties.reply_to, body='answer99')
    print('message sent')


connection_parameters = pika.ConnectionParameters('localhost')

connection = pika.BlockingConnection(connection_parameters)

channel = connection.channel()
channel.queue_declare(queue='request-queue')

channel.basic_consume(queue='request-queue', auto_ack=True,
                      on_message_callback=on_request_message_received)

print("Starting Server")

channel.start_consuming()


"""

url = 'https://drive.google.com/file/d/13YjZVeddq91-j1LTqu9DqEzZWPCOeq98/dkjkj'

#data = models.Datasets(dataset_link = url, rows_number = 10, columns_number=4, baseline=23.2)

crud.create_dataset(data, models.SessionLocal())
dataset = crud.get_dataset(models.SessionLocal(), url)
best_runid = dataset.baseline
print(best_runid)"""