import pika
import functools
import os
from retry import retry
import uuid
import models
from causal_discovery_algs import *
from sqlalchemy.orm import Session as db
n_splits =2
indep_test = [gsq, mv_fisherz, fisherz]
alpha = [0.05, 0.2, 0.8]
out = "."

def on_reply_message_received(ch, method, properties, body):
    with open('answer.txt', 'w') as f:
        f.write(body.decode('utf8'))

def check_readiness():
    #while ready == '0':
       with open('ready.txt', 'r') as f:
           ready = f.read()
       return ready

def consume_messages():
    channel.start_consuming()

def get_result(url, old_is_better=False, new_results=False):
    # проверяем, делали ли мы расчет для датасета, сохраненного по данной ссылке
    data = pd.read_csv(url)
    try:
        dataset = crud.get_dataset(models.SessionLocal(), url)
        best_runid = dataset.best_run
        benchmark_mean = dataset.baseline


        print('got dataset')

        best_run = crud.get_best_run(models.SessionLocal(), best_runid)
        print('got best run')
        old_methods, old_alphas, old_launch_results, old_launch_best_method, old_launch_best_alpha, old_best_params = best_run.methods, \
        best_run.alphas, best_run.best_result, best_run.best_method, best_run.best_alpha, best_run.best_params

    # проверяем, появились ли с момента последнего запуска какие-либо изменения в наборе гиперпараметров.
    # если появились, запустим отдельно для них поиск лучших и сравним с предыдущим лучшим результатом
        new_methods = list(set(indep_test) - set(old_methods))
        new_alphas = list(set(alpha) - set(old_alphas))

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
            old_params=old_best_params,
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
                                                                               old_params=old_best_params,
                                                                               benchmark_mean=benchmark_mean,
                                                                               old_launch_best_method=old_launch_best_method,
                                                                               old_launch_best_alpha=old_launch_best_alpha)
            print('got answer')


        #if not old_is_better:
          #crud.update_best_run(models.SessionLocal(), dataset_id=dataset.id, best_runid=run.id)
        #return answer


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
       print(dataset.id)
       #dataset = crud.get_dataset(models.SessionLocal(), url)
       #print('got dataset')

       run = models.Runs(
             dataset_id=dataset.id,
             best_result=best_launch,
             best_algorithm='PC',
             best_method=best_method,
             best_alpha=best_method,
             alphas=alpha,
             methods=indep_test)
       crud.create_run(run, models.SessionLocal())
       print(run.id)

    if not old_is_better:
          crud.update_best_run(models.SessionLocal(), dataset_id=dataset.id, best_runid=run.id)
    return answer

def get_db():
    db = models.SessionLocal()
    try:
        yield db
    finally:
        db.close()
class AmqpConnection:
    def __init__(self, hostname='localhost', port=5672, username='guest', password='guest'):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.connection = None
        self.channel = None

    def connect(self, connection_name='Neat-App'):
            print('Attempting to connect to', self.hostname)
            params = pika.ConnectionParameters(
                host=self.hostname,
                port=self.port,
                credentials=pika.PlainCredentials(self.username, self.password),
                client_properties={'connection_name': connection_name})
            self.connection = pika.BlockingConnection(parameters=params)
            self.channel = self.connection.channel()
            print('Connected Successfully to', self.hostname)
    def setup_queues(self):
        self.channel.exchange_declare('Ping_Exchange', exchange_type='direct')
        self.channel.queue_declare('Ping_Queue')
        self.channel.queue_bind('Ping_Queue', exchange='Ping_Exchange', routing_key='request-queue')
    def do_async(self, callback, *args, **kwargs):
        if self.connection.is_open:
            self.connection.add_callback_threadsafe(functools.partial(callback, *args, **kwargs))
    def publish(self, payload):
        if self.connection.is_open and self.channel.is_open:
            self.channel.basic_publish(
                exchange='Ping_Exchange',
                routing_key='request-queue',
                body=payload,
                properties=pika.BasicProperties(
                    reply_to=reply_queue.method.queue,
                    correlation_id=str(uuid.uuid4())))



    @retry(pika.exceptions.AMQPConnectionError, delay=1, backoff=2)
    def consume(self, on_message):
        if self.connection.is_closed or self.channel.is_closed:
            self.connect()
            self.setup_queues()
        try:
            self.channel.basic_consume(queue='Ping_Queue', auto_ack=True, on_message_callback=on_message)
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print('Keyboard interrupt received')
            self.channel.stop_consuming()
            self.connection.close()
            os._exit(1)
        except pika.exceptions.ChannelClosedByBroker:
            print('Channel Closed By Broker Exception')
