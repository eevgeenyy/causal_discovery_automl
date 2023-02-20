from utils import AmqpConnection
from threading import Thread
import time
import random
from telebot.async_telebot import AsyncTeleBot
import asyncio
#import nest_asyncio
import pandas as pd
import pika
import config

messages = []
msg_counts = {}

bot = AsyncTeleBot(token=config.token)

def send_message(url):
    mq.do_async(mq.publish, payload=url)

def on_message(body):
    with open('answer.txt', 'w') as f:
        f.write(body.decode('utf8'))

mq = AmqpConnection()
mq.connect()
mq.setup_queues()


@bot.message_handler(func=lambda message: True)
async def echo_all(message):
    url = message.text
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    await bot.reply_to(message, 'получил файл, работаем')
    try:
        Thread(target=send_message(url), args=[mq]).start()
        mq.consume(on_message)
        with open('answer.txt', 'r') as f:
            answer = f.read()
        await bot.reply_to(message, answer)
    except Exception as e:
        await bot.reply_to(message,
                           'какая-то ошибка :( проверьте, что файл в формате csv и содержит только категориальные значения. Файл должен лежать на GDrive')
        logging.error(traceback.format_exc())


asyncio.run(bot.infinity_polling(True))
