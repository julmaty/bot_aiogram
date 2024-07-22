import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.utils.formatting import (
    Bold, as_list, as_marked_section, as_key_value, HashTag
)

from config_reader import config

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=config.bot_token.get_secret_value(), default=DefaultBotProperties(
        parse_mode=ParseMode.HTML
    ))
dp = Dispatcher()


# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    kb = [
        [types.KeyboardButton(text="С пюрешкой"),
         types.KeyboardButton(text="Без пюрешки")]
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder="Выберите способ подачи")
    await message.answer("Как подавать котлеты?", reply_markup=keyboard)

@dp.message(F.text.lower() == "с пюрешкой")
async def with_puree(message: types.Message):
    await message.reply("Отличный выбор!", reply_markup=types.ReplyKeyboardRemove())

@dp.message(F.text.lower() == "без пюрешки")
async def without_puree(message: types.Message):
    await message.reply("Так невкусно!", reply_markup=types.ReplyKeyboardRemove())

@dp.message(Command("help"))
async def process_help_command(message: types.Message):
    await message.answer("Напиши мне что-нибудь, и я отпрпавлю этот текст тебе в ответ!")

@dp.message(Command("hello"))
async def cmd_hello(message: types.Message):
    await message.answer(
        f"Hello, <b>{message.from_user.full_name}</b>"
    )

@dp.message(F.text, Command("test"))
async def any_message(message: types.Message):
    await message.answer(
        "Hello, <b>world</b>!"
    )

@dp.message(Command("advanced_example"))
async def cmd_advanced_example(message: types.Message):
    content = as_list(
        as_marked_section(
            Bold("Success:"),
            "Test 1",
            "Test 3",
            "Test 4",
            marker="✅ ",
        ),
        as_marked_section(
            Bold("Failed:"),
            "Test 2",
            marker="❌ ",
        ),
        as_marked_section(
            Bold("Summary:"),
            as_key_value("Total", 4),
            as_key_value("Success", 3),
            as_key_value("Failed", 1),
            marker="  ",
        ),
        HashTag("#test"),
        sep="\n\n",
    )
    await message.answer(**content.as_kwargs())

@dp.message(F.photo)
async def download_photo(message: types.Message, bot: Bot):
    await bot.download(
        message.photo[-1],
        destination=f"tmp/{message.photo[-1].file_id}.jpg"
    )


@dp.message(F.sticker)
async def download_sticker(message: types.Message, bot: Bot):
    await bot.download(
        message.sticker,
        # для Windows пути надо подправить
        destination=f"tmp/{message.sticker.file_id}.webp"
    )

@dp.message(F.animation)
async def echo_gif(message: types.Message):
    await message.reply_animation(message.animation.file_id)

@dp.message()
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, msg.text)


# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

