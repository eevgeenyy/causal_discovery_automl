import time
from telebot.async_telebot import AsyncTeleBot
import asyncio
import traceback
import logging
import config
import uuid
from utils import *
from subprocess import Popen
import sys
#path = 'C:/Users/Evgeny/Documents/GitHub/causal_discovery_automl/venv/Scripts/python.exe'
#path = '/usr/local/bin/python3'
#Popen([path, 'models.py'], shell = True)
#Popen([path, 'server.py'], shell = True)
#exec(open("./models.py").read())
#exec(open("./server.py").read())
bot = AsyncTeleBot(token=config.token)
greeting = "Это чат-бот - интерфейс к программе для автоматического подбора методов и гиперпараметров для поиска причинного графа. Вы можете направить ссылку на GDrive файл в формате csv и мы попробуем подобрать для него лучший набор гиперпараметров"

#connect_params = pika.ConnectionParameters('localhost')
credentials = pika.PlainCredentials('guest', 'guest')
connect_params = pika.ConnectionParameters('rabbitmq3', 5672, '/', credentials, heartbeat=600, blocked_connection_timeout=300)

connection = pika.BlockingConnection(connect_params)

channel = connection.channel()

channel.queue_declare(queue='request-queue')
print('passed', flush=True)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
        bot.reply_to(message, greeting)
@bot.message_handler(func=lambda message: True)
async def echo_all(message):
    cor_id = str(uuid.uuid4())
    url = message.text
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    await bot.reply_to(message, 'получил файл, работаем')
    try:
        channel.basic_publish('', routing_key='request-queue', properties=pika.BasicProperties(
                correlation_id=cor_id
            ), body=url)
        print('message sent', flush=True)

        ready = "0"
        while ready == '0':
            ready = check_readiness()
            time.sleep(10)
        f = open(f'{cor_id}.txt', encoding="utf8")
        answer = f.read()
        f.close()
        print(answer)
        await bot.reply_to(message, answer)
        with open('ready.txt', 'w') as f:
            f.write('0')
        connection.close()

    except Exception as e:
           await bot.reply_to(message,
                           'какая-то ошибка :( проверьте, что файл в формате csv и содержит только категориальные значения. Файл должен лежать на GDrive')
           logging.error(traceback.format_exc())


asyncio.run(bot.infinity_polling(True))






