import time
import pika

from utils import *
def on_request_message_received(ch, method, properties, body):
    print(f"Received Request: {properties.correlation_id}")
    answer = get_result(body.decode('utf8'))
    print(answer, flush=True)
    with open(f'{properties.correlation_id}.txt', 'w', encoding='utf-8') as f:
        f.write(answer)

    with open('ready.txt', 'w') as f:
        f.write('1')
    print('message sent')

print("opened Server", flush=True)
#connection_parameters = pika.ConnectionParameters('localhost')
credentials = pika.PlainCredentials('guest', 'guest')
#connect_params = pika.ConnectionParameters('localhost')
connect_params = pika.ConnectionParameters('rabbitmq3', 5672, '/', credentials)
time.sleep(1)
connection = pika.BlockingConnection(connect_params)

channel = connection.channel()
time.sleep(1)
channel.queue_declare(queue='request-queue')
time.sleep(1)
channel.basic_consume(queue='request-queue', auto_ack=True,
                      on_message_callback=on_request_message_received)

print("Starting Server", flush=True)

channel.start_consuming()
