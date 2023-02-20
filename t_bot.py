import threading
import telebot
import re
from telebot import types
import time
import csv
import string
from telebot.async_telebot import AsyncTeleBot
import asyncio
#import nest_asyncio
import pandas as pd
import pika
import traceback
import logging
import config
import os
import uuid

from collections import defaultdict
#nest_asyncio.apply()
#bot = AsyncTeleBot(token=os.getenv('token'))

bot = AsyncTeleBot(token=config.token)

greeting = "Это чат-бот - интерфейс к программе для автоматического подбора методов и гиперпараметров для поиска причинного графа. Вы можете направить ссылку на GDrive файл в формате csv и мы попробуем подобрать для него лучший набор гиперпараметров"

def on_reply_message_received(ch, method, properties, body):
    with open('answer.txt', 'w') as f:
        f.write(body.decode('utf8'))

def check_readiness(ready = 0):
    while ready == '0':
       with open('ready.txt', 'r') as f:
           ready = f.read()
           time.sleep(30)
       f.close()

def restart():
    with open('ready.txt', 'w') as f:
        f.write('0')
def consume_messages():
    channel.start_consuming()

connect_params = pika.ConnectionParameters('localhost', heartbeat=600, blocked_connection_timeout=300)

connection = pika.BlockingConnection(connect_params)

channel = connection.channel()

#reply_queue = channel.queue_declare(queue='', exclusive=True)

#channel.basic_consume(queue=reply_queue.method.queue, auto_ack=True,
 #                     on_message_callback=on_reply_message_received)

channel.queue_declare(queue='request-queue')



#channel.start_consuming()
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
        bot.reply_to(message, greeting)
@bot.message_handler(func=lambda message: True)
async def echo_all(message):
    #global message
    #message = mes
    cor_id = str(uuid.uuid4())
    url = message.text
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    await bot.reply_to(message, 'получил файл, работаем')
    try:
        #channel.basic_publish(exchange='', routing_key='', body=url)
        #if connection.is_closed:
            #channel = connection.channel()
           # channel.queue_declare(queue='request-queue')
        print(channel)


        channel.basic_publish('', routing_key='request-queue', properties=pika.BasicProperties(
                #reply_to=reply_queue.method.queue,
                correlation_id=cor_id
            ), body=url)
        print('message sent')

        #threading.Thread(target=consume_messages, daemon=True).start()
        check_readiness()
        time.sleep(3)

        with open(f'{cor_id}.txt', 'r') as f:
            answer = f.read()
        print(answer)
        await bot.reply_to(message, answer)
        restart()

        connection.close()


         #   print(f"received new message: {body}")


    except Exception as e:
           await bot.reply_to(message,
                           'какая-то ошибка :( проверьте, что файл в формате csv и содержит только категориальные значения. Файл должен лежать на GDrive')
           logging.error(traceback.format_exc())


asyncio.run(bot.infinity_polling(True))






