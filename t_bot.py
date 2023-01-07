
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
import traceback
import logging
import os

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import gsq
from causallearn.utils.cit import mv_fisherz

from collections import defaultdict
from causal_discovery_algs import *

#nest_asyncio.apply()
bot = AsyncTeleBot(token=os.getenv('token'))
greeting = "Это чат-бот - интерфейс к программе для автоматического подбора методов и гиперпараметров для поиска причинного графа. Вы можете направить ссылку на GDrive файл в формате csv и мы попробуем подобрать для него лучший набор гиперпараметров"
indep_test = [gsq, mv_fisherz, fisherz]
alpha = [0.05, 0.2, 0.8]
n_splits = 2
out = "C:/Users/Evgeny/PycharmProjects/pythonProject"


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, greeting)


@bot.message_handler(func=lambda message: True)
async def echo_all(message):
    try:
        url = message.text
        url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
        data = pd.read_csv(url)
        await bot.reply_to(message, 'получил файл, работаем')
        benchmark, benchmark_mean, methods, alphas, launch_results, launch_metrics, launch_params = find_best_params(
            data, alpha, indep_test, n_splits)
        best_result = min(launch_results, key=launch_results.get)
        best_launch = launch_results[best_result]
        best_params = launch_params[best_result]
        if best_launch < benchmark_mean:
            cg = pc(data.to_numpy(), alpha=alphas[best_result], indep_test=methods[best_result], stable=True, uc_rule=0,
                    uc_priority=-1)
            answer = f'Ура! Для ваших данных был подобран каузальный граф. Используемый метод - PC, параметры: {best_params}, среднее MSE - {best_result}, MSE бенчмарка - {benchmark_mean}'
            pyd = GraphUtils.to_pydot(cg.G, labels=data.columns)
            pyd.write_png('graph.png')
            await bot.send_photo(message.from_user.id, open("graph.png", 'rb'))
            os.remove("graph.png")
        else:
            answer = f'К сожалению, для ваших данных не удалось найти подходящий причинный граф. Используемый метод - PC, параметры лучшего результата: {best_params}, MSE лучшего результата - {best_launch}, MSE бенчмарка - {benchmark_mean}'
        await bot.reply_to(message, answer)

    except Exception as e:
        await bot.reply_to(message,
                           'какая-то ошибка :( проверьте, что файл в формате csv и содержит только категориальные значения. Файл должен лежать на GDrive')
        logging.error(traceback.format_exc())


asyncio.run(bot.infinity_polling(True))






