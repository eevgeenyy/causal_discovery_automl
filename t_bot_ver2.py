import time
from telebot.async_telebot import AsyncTeleBot
import asyncio
import traceback
import config
import uuid
from utils import *
from subprocess import Popen
from threading import Thread
import sys
import logging
logging.basicConfig(level=logging.DEBUG)
import aiohttp
import aiogram
port = 5672
username = "guest"
password = "guest"
exchangeName = "Ping_Exchange"
replyQueue = "replyQ"
requestKey = "request"
replyKey = "reply"
bot = AsyncTeleBot(token=config.token)
greeting = "Это чат-бот - интерфейс к программе для автоматического подбора методов и гиперпараметров для поиска причинного графа. Вы можете направить ссылку на GDrive файл в формате csv и мы попробуем подобрать для него лучший набор гиперпараметров"
channel = None
def on_responce(channel, method, properties, body):
    print(body)
    #close connection once receives the reply
    status.answer = body.decode('utf8')
    channel.stop_consuming()
    #connection.close()
    status.answer_ready = True

def listen(connection, channel):
    print('Entering listen function...')
    channel.queue_declare(queue=replyQueue, exclusive=True, auto_delete=True)
    channel.queue_bind(exchange=exchangeName, queue=replyQueue, routing_key=replyKey)
    channel.basic_consume(on_message_callback=lambda ch, method, properties, body: on_responce(channel, method, properties, body), queue=replyQueue)
    print('Start consuming messages...')
    channel.start_consuming()
    connection.close()
class StatusChecker:
    def __init__(self):
        self.answer_ready = False
        self.answer = None
        self.url = None

status = StatusChecker()


async def polling():
    logging.debug('Polling function started')
    async with aiohttp.ClientSession() as session:
        bot2 = aiogram.Bot(token=config.token)
        bot2._session = session
        dp = aiogram.dispatcher.Dispatcher(bot2)
        dp.skip_updates = True

        @dp.message_handler()
        async def echo(message: aiogram.types.Message):
            print('message.text')
            await message.reply("echo")
        await dp.start_polling()

async def other_task():
    global channel
    while True:
        print('ok')
        while status.url is None:
            pass
        print('go!')
        try:
            credentials = pika.PlainCredentials(username, password)
            connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            channel = connection.channel()
            Thread(target=listen()).start()
            time.sleep(1)  # give time for it to start consuming
            # send request message
            properties = pika.spec.BasicProperties(content_type="text/plain", delivery_mode=1, reply_to=replyKey)
            channel.basic_publish(exchange=exchangeName, routing_key=requestKey, body=status.url,
                                  properties=properties)
            print('message sent')
            while connection.is_open:
                pass
            status.answer_ready = False
            status.answer = None
            status.url = None
        except Exception as e:
            print("Error:", e)

async def main():
    # create a task for polling and other task
    polling_task = asyncio.create_task(polling())
    other_tasks = asyncio.create_task(other_task())

    # wait for polling and other task to complete
    await asyncio.gather(polling_task, other_tasks)

asyncio.run(main(), debug=True)








