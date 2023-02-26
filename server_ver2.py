from utils import AmqpConnection
from threading import Thread
import pika
import time

port = 5672
username = "guest"
password = "guest"
exchangeName = "Ping_Exchange"
requestQueue = "requestQ"
replyQueue = "replyQ"
requestKey = "request"
replyKey = "reply"

def on_message(channel, method, properties, body):
    print("Received request: %r" % body)
    try:
        replyProp = pika.BasicProperties(content_type="text/plain", delivery_mode=1)
        channel.basic_publish(exchange=exchangeName, routing_key=properties.reply_to, properties=replyProp,
                              body="Reply to %s" % body)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except:
        channel.basic_nack(delivery_tag=method.delivery_tag)

while True:
    try:
        #connect
        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        channel = connection.channel()

        #declare exchange and queue, bind them and consume messages
        channel.exchange_declare(exchange = exchangeName, exchange_type = "direct")
        channel.queue_declare(queue = requestQueue, exclusive = True)
        channel.queue_bind(exchange = exchangeName, queue = requestQueue, routing_key = requestKey)
        channel.basic_consume(on_message_callback = on_message, queue = requestQueue)
        print('server started')
        channel.start_consuming()
        print('server started')
    except Exception as e:
        #reconnect on exception
        print("Exception handled, reconnecting...\nDetail:\n%s" % e)
        try:
            connection.close()
        except:
            pass
        time.sleep(5)

