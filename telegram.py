from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import asyncio
import aioschedule
import pandas as pd

import graphics
import strategy_final
import subs

API_TOKEN = 'token'

bot = Bot(token=API_TOKEN)
# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


# States
class Form(StatesGroup):
    ticker = State()
    start = State()
    stop = State()
    mva = State()
    graph_type = State()
    tik = State()
    mva2 = State()
    randomd = State()
    start2 = State()
    end = State()
    pattern = State()
    ticker3 = State()
    start3 = State()
    stop3 = State()
    subs_ticker = State()
    sma=State()


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await bot.send_message(message.from_user.id, text='Привет! Для начала работы ознакомьтесь с доступными командами.\n\n💡/start\nВозвращает Вас к полному списку команд и их возможностей.\n\n📈 /candles_plot\nБот запускает построение графика, выбранной Вами акции на фондовом рынке. Для изменения тикера также вызовите данную команду\n\n 📊 /plot_with_signal\nДля того, чтобы увидеть график цен закрытия с сигналами для покупки или продажи в выбранный вами период.')
    await bot.send_message(message.from_user.id, text='💼 /try_strategy\nДля того, чтобы оценить работу стратегии (SMA) на основе данных прошлых лет строится  аналитическая таблица, где Вы можете увидеть эффективность реализуемой стратегии.\n\n📩 /subscribe\nДля того, чтобы подписаться на ежедневную рассылку вызовите данную команду.\n\n🤝 /help\nЕсли возникнут технические неполадки или другие вопросы по работе бота можете обратиться по этой ссылке:  @yolomiQ  и мы вам поможем!😉')


@dp.message_handler(commands=['candles_plot'])
async def handle_text(message: types.Message):
    try:
        await Form.ticker.set()
        await bot.send_message(message.from_user.id, text='1. Введите интересующий Вас тикер:\n\n1) xxxx, где xxxx - индекс акции NASDAQ, например YNDX.\n\n2) xxxx.me, где xxxx - индекс акции MOEX c дополнением (.me), например SBER.me',
                               parse_mode='HTML')
    except:
        print('Ошибка в части candles_plot2')


@dp.message_handler(state=Form.ticker)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['ticker'] = message.text
        await Form.start.set()
        await bot.send_message(message.from_user.id, text="2. Введите начало периода, который хотите посмотреть:\n\n хххх-хх-хх, гггг-мм-дд\n Например 2020-01-01",
                               parse_mode='HTML')
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


@dp.message_handler(state=Form.start)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['start'] = message.text
        await Form.stop.set()
        await bot.send_message(message.from_user.id, text="3. Введите конец периода, который хотите посмотреть:\n\n хххх-хх-хх, гггг-мм-дд\n Например 2020-02-01",
                               parse_mode='HTML')
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


@dp.message_handler(state=Form.stop)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['stop'] = message.text
        await Form.graph_type.set()
        await bot.send_message(message.from_user.id, text="4. Введите тип графика:\n\n🔸candle - график, сочетающий в себе японские свечи и объем торговли. Показывает поведение цены на вертикальной оси и время на горизонтальной.\n\n🔸line - для построения такого графика берётся цена закрытия за выбранный период.\n\n🔸renko - график отображает тренд (изменение цены). Данный график также подаёт сигналы о разворотах и серьезных предостережениях торговли.\n\n🔸pnf - график показывает те движения цены, которые превышают установленный размер.",
                               parse_mode='HTML')
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


@dp.message_handler(state=Form.graph_type)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['graph_type'] = message.text
        await Form.mva.set()
        await bot.send_message(message.from_user.id, text="5. Введите значение индикатора(SMA): формат ввода периода для двух скользящих средних «х,х»\n Например 5,10",
                               parse_mode='HTML')
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


@dp.message_handler(state=Form.mva)
async def process_gender(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['mva'] = message.text
            mva = tuple(int(i) for i in list(data['mva'].split(',')))
        graphics.plot_japan(data['ticker'], data['start'], data['stop'], data['graph_type'], mva)
        await bot.send_photo(message.from_user.id, open('graphs/japan.png', 'rb'))
        await bot.send_message(message.from_user.id, text='🔔Ежедневно в 8:30 вы можете получать уведомления об изменении состояния цены акции и предложение о покупке или продажи акции(а также о режиме ожидания)\n✔️Для подписки выполните команду /subscribe')
        await bot.send_message(message.from_user.id, text='Если Вы хотите посмотреть  другой тикер вызовите команду /candle_plot')
        await state.finish()
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await bot.send_message(message.from_user.id,text="Если возникнут технические неполадки\
                                                      или другие вопросы по работе бота \
                                                      можете обратиться по этой ссылке: \
                                                      @yolomiQ  и мы вам поможем!😉")


@dp.message_handler(commands=['plot_with_signal'])
async def handle_text(message: types.Message):
    try:
        await Form.tik.set()
        await bot.send_message(message.from_user.id, text='1. Введите интересующий Вас тикер:\n\n1) xxxx, где xxxx - индекс акции NASDAQ, Например YNDX.\n\n2) xxxx.me, где xxxx - индекс акции MOEX c дополнением (.me), Например SBER.me',

                               parse_mode='HTML')
    except:
        print('Ошибка в части plot_with_signal')


@dp.message_handler(state=Form.tik)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['ticker'] = message.text
        await Form.start2.set()
        await bot.send_message(message.from_user.id, text="2. Введите начало периода, который хотите посмотреть:\n\n хххх-хх-хх, гггг-мм-дд\n Например 2020-01-01",
                               parse_mode='HTML')
    except:
        await state.finish()
        await bot.send_message(message.from_user.id, text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')

@dp.message_handler(state=Form.start2)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['start2'] = message.text
        await Form.end.set()
        await bot.send_message(message.from_user.id, text="3. Введите конец периода, который хотите посмотреть:\n\n хххх-хх-хх, гггг-мм-дд\n Например 2020-02-01",
                               parse_mode='HTML')
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


@dp.message_handler(state=Form.end)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['end'] = message.text
        await Form.mva2.set()
        await bot.send_message(message.from_user.id,
                               text="5. Введите значение индикатора(SMA): формат ввода периода для двух скользящих средних «х,х»\n Например 5,10",
                               parse_mode='HTML')
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


@dp.message_handler(state=Form.mva2)
async def process_gender(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['mva2'] = message.text
            mva = tuple(int(i) for i in list(data['mva2'].split(',')))
        graphics.plot_with_signal(data['ticker'], data['start2'], data['end'], mva)
        await bot.send_photo(message.from_user.id, open('graphs/plot_with_signal.png', 'rb'))
        await bot.send_message(message.from_user.id,
                               text='🔔Ежедневно в 8:30 вы можете получать уведомления об изменении состояния цены акции и предложение о покупке или продажи акции(а также о режиме ожидания)\n✔️Для подписки выполните команду /subscribe')
        await bot.send_message(message.from_user.id,
                               text='Если Вы хотите посмотреть  другой тикер вызовите команду /candle_plot')
        await state.finish()
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


####Backtesting####
@dp.message_handler(commands=['try_strategy'])
async def handle_text(message: types.Message):
    try:
        await Form.ticker3.set()
        await bot.send_message(message.from_user.id,
                               text='1. Введите интересующий Вас тикер:\n\n1) xxxx, где xxxx - индекс акции NASDAQ, Например YNDX.\n\n2) xxxx.me, где xxxx - индекс акции MOEX c дополнением (.me), Например SBER.me',
                               parse_mode='HTML')
    except:
        print('Ошибка в части try_strategy')


@dp.message_handler(state=Form.ticker3)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['ticker'] = message.text
        await Form.start3.set()
        await bot.send_message(message.from_user.id, text="2. Введите начало периода, который хотите посмотреть:\n\n хххх-хх-хх, гггг-мм-дд\n Например 2020-01-01",
                               parse_mode='HTML')
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


@dp.message_handler(state=Form.start3)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['start'] = message.text
        await Form.stop3.set()
        await bot.send_message(message.from_user.id, text="3. Введите конец периода, который хотите посмотреть:\n\n хххх-хх-хх, гггг-мм-дд\n Например 2020-02-01",
                               parse_mode='HTML')
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


@dp.message_handler(state=Form.stop3)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['stop'] = message.text
        strategy_final.try_strategy(data['ticker'], data['start'], data['stop'])
        await bot.send_photo(message.from_user.id, open('graphs/result.png', 'rb'))
        await bot.send_message(message.from_user.id,
                               text='🔔Ежедневно в 8:30 вы можете получать уведомления об изменении состояния цены акции и предложение о покупке или продажи акции(а также о режиме ожидания)\n✔️Для подписки выполните команду /subscribe')
        await bot.send_message(message.from_user.id,
                               text='Если Вы хотите посмотреть  другой тикер вызовите команду /candle_plot')
        await state.finish()
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


###Подписка###
@dp.message_handler(commands=['subscribe'])
async def handle_text(message: types.Message):
    try:
        await Form.subs_ticker.set()
        await bot.send_message(message.from_user.id,
                               text="На какой тикер(ы) Вы бы хотели подписаться?\n\nВведите интересующий Вас тикер:\n1) xxxx, где xxxx - индекс акции NASDAQ, Например YNDX.\n2) xxxx.me, где xxxx - индекс акции MOEX c дополнением .me, Например SBER.me\n\n Если же вы хотите подписаться на несколько тикеров, то вводите их через запятую, слитно\nНапример AAPL,SBER.me",
                               parse_mode='HTML')
    except:
        print('Ошибка в части subscribe')


@dp.message_handler(state=Form.subs_ticker)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['subs_ticker'] = message.text
            data['username'] = message.from_user.id
        await Form.sma.set()
        await bot.send_message(message.from_user.id,
                               text="Введите значение индикатора(SMA): формат ввода периода для двух скользящих средних «х,х»\n Например 5,10",
                               parse_mode='HTML')
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


@dp.message_handler(state=Form.sma)
async def process_name(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            data['sma'] = message.text
        subs.set_tickers(data['username'], data['subs_ticker'], data['sma'])
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Тикер(ы) успешно добавлен(ы)",
                               parse_mode='HTML')
    except:
        await state.finish()
        await bot.send_message(message.from_user.id,
                               text="Что-то пошло не так, попробуйте еще раз\nОбратите внимание, что тикеры русской биржи должны быть введены с добавлением .me в конце\nПример\nSBER.me",
                               parse_mode='HTML')


###Отправка индикаторов###
async def send_indicator():
    try:
        strategy_final.comp_strategy()
        df = pd.read_csv('database.csv', header=0, sep=';')
        for index, row in df.iterrows():
            await bot.send_message(row.nickname, text='Тикер:{0}\n Действие:{1}'.format(row.tickers, row.actions_all))
    except:
        print('Ошибка в send_indicator()')


async def scheduler():
    aioschedule.every().day.at("8:30").do(send_indicator)
    while True:
        await aioschedule.run_pending()
        await asyncio.sleep(1)


async def on_startup(_):
    asyncio.create_task(scheduler())


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=False, on_startup=on_startup)
