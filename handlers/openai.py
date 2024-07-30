from aiogram import Bot, Router, types, F
from aiogram.filters.command import Command
from aiogram.client.default import DefaultBotProperties
from pinecone import Pinecone, ServerlessSpec
from config_reader import config

from openai import OpenAI

router = Router()
tokenAI=config.openai_key_personal.get_secret_value()
tokenPC=config.pinecone_key.get_secret_value()
client = OpenAI(
  api_key=tokenAI
)
pc = Pinecone(api_key=tokenPC)



@router.message(Command("help"))
async def process_help_command(message: types.Message):
    await message.answer("Напиши мне что-нибудь, и я отпрпавлю этот текст тебе в ответ!")

@router.message(Command("ideas"))
async def ideas(message: types.Message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты креативный менеджер в компьютерной компании"},
            {"role": "user", "content": "Придумай 10 идей для программы помощника студентам"},
        ]
    )
    res = response.choices[0].message.content

    await message.answer(res)

@router.message(Command("codeC"))
async def code(message: types.Message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты программист, пишущий на языке C# c использованием фреймворка ASP.Net Core"},
            {"role": "user", "content": "Напиши API для сервиса Помощника по организации стажировок и практик"},
        ]
    )
    res = response.choices[0].message.content

    await message.answer(f"{res}", parse_mode=None)

@router.message(Command("codeC"))
async def codeFront(message: types.Message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты программист, пишущий на языке C# c использованием фреймворка ASP.Net Core"},
            {"role": "user", "content": "Напиши API для сервиса Помощника по организации стажировок и практик"},
        ]
    )
    res = response.choices[0].message.content

    await message.answer(f"{res}", parse_mode=None)


@router.message(Command("analisis"))
async def analisis(message: types.Message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты аналитик в команде программистов"},
            {"role": "user", "content": "Напиши юзер-кейсы для сервиса Помощник по организации стажировок и практик"},
        ]
    )
    res = response.choices[0].message.content

    await message.answer(res)

@router.message(Command("presentation"))
async def presentation(message: types.Message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты креативный аналитик в команде программистов"},
            {"role": "user", "content": "Напиши план презентации сервиса Помощник по организации стажировок и практик. План должен включать в себя задачи, которые решает сервис, его актуальность, перспективы развития сервия и другие пункты. Формулировки должны быть продающими, сформулированы креативно"},
        ]
    )
    res = response.choices[0].message.content

    await message.answer(res)


@router.message(Command("cat"))
async def echo_gif(message: types.Message):
    response = client.images.generate(
        model="dall-e-2",
        prompt="a white siamese cat",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url

    await message.reply_photo(image_url)