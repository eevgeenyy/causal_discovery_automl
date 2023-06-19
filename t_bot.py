
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import traceback
import logging
from src.utils import *
from src.answers_preparation import *
from src.estimator import *
from src.db_quiries_functions import *
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, Message, ReplyKeyboardRemove
from aiogram.dispatcher.filters import Text


def read_token_from_file(file_path):
    with open(file_path, 'r') as file:
        token = file.readline().strip()
    return token


file_path = 'token.txt'
token = read_token_from_file(file_path)


bot = Bot(token=token)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
class BotStates(StatesGroup):
    data = State()
    datatype = State()
    preferences = State()



greeting = "Это чат-бот - интерфейс к программе для автоматического подбора методов и гиперпараметров для поиска причинного графа. Вы можете направить ссылку на GDrive файл в формате csv и мы попробуем подобрать для него лучший набор гиперпараметров"
data_type_question = 'Пожалуста, уточните тип значений переменных в датасете. Для категориальных переменных поддерживаются бинарные значения'
data_type_keyboard = ['Дискретные', 'Категориальные', 'Вещественные']
preferences_question = 'Есть ли у вас предпочтения относительно типа возможных ошибок?'
preferences_keyboard = ['Лучше, если будут отсутствовать некоторые существующие связи',
                            'Лучше, если будет присутствовать связи, которых на самом деле нет',
                            'Нет предпочтений']

@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='отмена', ignore_case=True), state='*')
async def cancel_handler(message, state: FSMContext):

    #Allow user to cancel any action

    current_state = await state.get_state()
    if current_state is None:
        return

    logging.info('Cancelling state %r', current_state)
    # Cancel state and inform user about it
    await state.finish()

    await message.reply('Запрос отменен. Вы можете начать сессию заново', reply_markup=ReplyKeyboardRemove())
@dp.message_handler(lambda message: True, state=None)
async def start(message: Message):
    await BotStates.data.set()
    await message.reply(greeting)


@dp.message_handler(lambda message: True, state=BotStates.data)
async def load_data(message: Message, state=FSMContext):
    async with state.proxy() as user_data:
        user_data['url'] = message.text
    await BotStates.next()

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(data_type_keyboard[0])
    markup.add(data_type_keyboard[1])
    markup.add(data_type_keyboard[2])

    await message.reply(data_type_question, reply_markup=markup)


@dp.message_handler(lambda message: True, state=BotStates.datatype)
async def set_datatype(message: Message, state=FSMContext):

    async with state.proxy() as user_data:
        user_data['data_type'] = message.text
    print(user_data['data_type'])
    await BotStates.next()

    ReplyKeyboardRemove()

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(preferences_keyboard[0])
    markup.add(preferences_keyboard[1])
    markup.add(preferences_keyboard[2])
    await message.reply(preferences_question, reply_markup=markup)

@dp.message_handler(lambda message: True, state=BotStates.preferences)
async def set_pref(message: Message, state=FSMContext):
        async with state.proxy() as user_data:
            user_data['preferences'] = message.text

        await message.reply("ok", reply_markup=ReplyKeyboardRemove())
        url = user_data['url']
        url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

        cache_file = f"{hash(url)}.json"  # .json file

        metric_type = choose_metric(user_data['data_type'])
        try:
            data = pd.read_csv(url)
            new_params, status, prev_launch_info = check_history(url, user_data['data_type'], user_data['preferences'])
            #status = 2

            if status == 0:
                old_methods, old_alphas, old_launch_results, old_launch_best_method, old_launch_best_alpha, uc_rule, uc_priority, benchmark = prev_launch_info
                await message.reply(f'для данного датасета и параметров запуска ранее были найдены следующие параметры лучшего датасета. \
                                   параметры: лучший метод {old_launch_best_method}, лучшее значение alpha {old_launch_best_alpha}, метрика {metric_type}, значение метрики - {old_launch_results}, значение бенчмарка - {benchmark}')

                cg = pc(data.to_numpy(), alpha=old_launch_best_alpha, indep_test=old_launch_best_method, uc_rule=uc_rule, uc_priority=uc_priority)
            elif status == 1:
                old_methods, old_alphas, old_launch_results, old_launch_best_method, old_launch_best_alpha, uc_rule, uc_priority, benchmark = prev_launch_info
                await message.reply(f'для данного датасета проводился расчет с другими параметрами запуска, методы {old_methods[0]}, значения alpha {old_alphas[0]}. Проводим новый запуск')

                new_best_alpha, new_launch_results, new_launch_metrics, new_launch_params = choose_alpha(data, new_params[1], old_launch_best_method, user_data['data_type'])
                metric_2, best_alpha, _ = unpack_best_launch(new_launch_results, new_launch_metrics, metric_type)
                alfa_msg, best_alpha = step2_answer(best_alpha, old_launch_best_alpha, metric_type, metric_2, old_launch_results)

                await message.reply(alfa_msg)
                cg = pc(data.to_numpy(), alpha=best_alpha, indep_test=old_launch_best_method, uc_rule=uc_rule, uc_priority=uc_priority)

            else:
                await message.reply('для данного датасета ранее расчет не проводился, запускаем...')
                alphas = get_alphas_from_prefs(user_data['preferences'])
                #step 1
                if user_data['data_type'] == 'Дискретные':
                    test_type = fisherz
                    _, benchmark = get_benchmark_cont(data)
                    new_launch_results, new_launch_metrics, new_launch_params = choose_integer_test(data, test_type, cache_file)
                    best_test, metric_1, counter = unpack_best_launch(new_launch_results, new_launch_metrics, metric_type)
                    await message.reply(f'Используем тест: {best_test}, значение MSE {metric_1}, значение метрики для бейзлайна - {benchmark},  выбираем лучшее значение alpha')


                elif user_data['data_type'] == 'Категориальные':
                    test_type = gsq
                    _, benchmark = get_benchmark_cat(data)

                    new_launch_results, new_launch_metrics, new_launch_params = choose_integer_test(data, test_type, cache_file)
                    best_test, metric_1, _ = unpack_best_launch(new_launch_results, new_launch_metrics, metric_type)
                    await message.reply(f'Лучший тест: {best_test}, значение коэф. Мэтью {metric_1}, значение метрики для бейзлайна - {benchmark}, выбираем лучшее значение alpha')


                else:
                    _, benchmark = get_benchmark_cont(data)
                    test_type = kci

                    launch_params, launch_results, kernel_type, power, width, launch_metrics = choose_cont_test(data, cache_file)
                    metric_1, best_result, counter = unpack_best_launch(launch_results, launch_metrics, metric_type)


                    await message.reply(f'Используем тест: KCI, значение MSE {metric_1}, значение метрики для бенчмарка - {benchmark}, тип ядра {kernel_type[counter]}, степень ядра {power[counter]}, ширина ядер X/Y {width[counter]}, выбираем лучшее значение alpha')
                #step2

                alphas_dict, launch_results, metric_2, _ = choose_alpha(data, alphas, test_type, user_data['data_type'])
                _, metric_2, best_result = unpack_best_launch(launch_results, alphas_dict, metric_type)
                best_alpha = alphas_dict[best_result]
                alfa_msg, best_alpha = step2_answer(best_alpha, 0.05, metric_type, metric_2, metric_1)
                await message.reply(alfa_msg)

                # step3
                uc_rule, uc_priority, metric_3 = choose_rules(data, test_type, best_alpha, user_data['data_type'])
                rules_msg, best_metric = step3_answer(uc_rule, uc_priority, metric_type, metric_3, metric_2)
                await message.reply(rules_msg)

                dataset_data = models.Datasets(
                    dataset_link=url,
                    rows_number = data.shape[0],
                    columns_number = data.shape[1],
                    baseline = benchmark)

                crud.create_dataset(dataset_data, models.SessionLocal())
                dataset = crud.get_dataset(url, models.SessionLocal())
                best_run_id = str(uuid.uuid4())



                if test_type != kci:
                    cg = pc(data.to_numpy(), alpha=best_alpha, indep_test=test_type, uc_rule=uc_rule, uc_priority=uc_priority)

                    run_data = models.Runs(dataset_id=dataset.id,
                                           run_id = best_run_id,
                                           best_result=best_metric,
                                           data_type=user_data['data_type'],
                                           best_algorithm='PC',
                                           best_test=test_type,
                                           best_alpha=best_alpha,
                                           methods=[test_type],
                                           alphas=alphas,
                                           uc_rule=uc_rule,
                                           uc_priority=uc_priority)



                elif kernel_type[counter] == 'Polynomial':
                  cg = pc(data.to_numpy(), alpha=best_alpha, indep_test=test_type, kernelX='Polynomial', kernelY='Polynomial', polyd=power[counter], uc_rule=uc_rule, uc_priority=uc_priority)
                  run_data = models.Runs(dataset_id=dataset.id,
                                         best_result=best_metric,
                                         data_type=user_data['data_type'],
                                         best_algorithm='PC',
                                         best_test=test_type,
                                         best_alpha=best_alpha,
                                         alphas=alphas,
                                         uc_rule=uc_rule,
                                         uc_priority=uc_priority,
                                         kernelX='Polynomial',
                                         kernelY='Polynomial',
                                         polyd=power[counter])


                elif kernel_type[counter] == 'Gaussian':
                  cg = pc(data.to_numpy(), alpha=best_alpha, indep_test=test_type, kernelX='Gaussian', kernelY='Gaussian', kwidthx=width[counter], kwidthy=width[counter], uc_rule=uc_rule, uc_priority=uc_priority)
                  run_data = models.Runs(dataset_id=dataset.id,
                                         best_result=best_metric,
                                         data_type=user_data['data_type'],
                                         best_algorithm='PC',
                                         best_test=test_type,
                                         best_alpha=best_alpha,
                                         alphas=alphas,
                                         uc_rule=uc_rule,
                                         uc_priority=uc_priority,
                                         kernelX='Gaussian',
                                         kernelY='Gaussian',
                                         kwidthx=width[counter],
                                         kwidthy=width[counter])
                else:
                  cg = pc(data.to_numpy(), alpha=best_alpha, indep_test=test_type, kernelX='Linear', kernelY='Linear', uc_rule=uc_rule, uc_priority=uc_priority)


            await message.reply(cg.G)
            await message.reply("Граф соотвествующий лучшим параметрам. Переменные пронумерованы в порядке очередности столбцов")
            if status == 2:
                crud.create_run(run_data, models.SessionLocal())
                crud.update_best_run(best_run_id, dataset.id, models.SessionLocal())

        except Exception as e:
               await message.reply('какая-то ошибка :( проверьте, что файл в формате csv и содержит только категориальные значения. Файл должен лежать на GDrive')
               logging.error(traceback.format_exc())
        await state.finish()

executor.start_polling(dp, skip_updates=True)







