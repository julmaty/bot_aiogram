from aiogram import Bot, Router, types, F
from aiogram.filters.command import Command
from aiogram.client.default import DefaultBotProperties
from config_reader import config

from openai import OpenAI

router = Router()
tokenAI=config.openai_key_personal.get_secret_value()
client = OpenAI(
  api_key=tokenAI
)



@router.message(Command("help"))
async def process_help_command(message: types.Message):
    await message.answer("Напиши мне что-нибудь, и я отпрпавлю этот текст тебе в ответ!")

@router.message(Command("assistant"))
async def echo_gif(message: types.Message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "user", "content": "Where was it played?"}
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