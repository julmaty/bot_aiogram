from aiogram import Bot, Router, types, F
from aiogram.filters.command import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.utils.formatting import (
    Bold, as_list, as_marked_section, as_key_value, HashTag
)
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from random import randint
from contextlib import suppress
from aiogram.exceptions import TelegramBadRequest
from typing import Optional
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.filters import StateFilter
from aiogram.filters.state import State, StatesGroup
from handlers.openai import name_ideas_Call, code_Call, codeFront_Call, analisis_Call, presentation_Call, tasks_utils, logo_Call

router = Router()

# Хэндлер на команду /start
@router.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.set_state(None)
    builder = ReplyKeyboardBuilder()

    builder.row(
        types.KeyboardButton(text="Идеи"),
        types.KeyboardButton(text="Код")
    )
    builder.row(
        types.KeyboardButton(text="Аналитика"),
        types.KeyboardButton(text="Дизайн")
    )

    builder.row(types.KeyboardButton(
        text="Задание на хакатон")
    )
    await message.answer("Какая задача интересует?", reply_markup=builder.as_markup(resize_keyboard=True))

@router.message(F.text.lower() == "задание на хакатон")
async def zadaniye(message: types.Message, state: FSMContext):
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = f"Текущее описание: \n{data['zadaniye_descr']} \n \nВы хотите указать другое задание?"
        builder = InlineKeyboardBuilder()
        builder.add(types.InlineKeyboardButton(
            text="Ввести",
            callback_data="yes_descr")
        )
        await message.answer(
            res,
            reply_markup=builder.as_markup()
        )     
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nВведите описание:"
        await state.set_state(Task_descr.opisaniye)
        await message.answer(res)

class Task_descr(StatesGroup):
    opisaniye = State()
    code_history = State()
    codeFront_history = State()
    ideas = State()
    tasks = State()

@router.callback_query(F.data == "yes_descr")
async def enter_descr(callback: types.CallbackQuery, state: FSMContext):   
    await callback.message.answer("Введите описание:")
    await state.set_state(Task_descr.opisaniye)
    await callback.answer()

@router.message(F.text.lower() == "идеи")
async def ideasChoice(message: types.Message, state: FSMContext):
    await state.set_state(None)
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="Идеи проекта",
        callback_data="ideas")
    )
    builder.add(types.InlineKeyboardButton(
        text="Идеи названия",
        callback_data="name_ideas")
    )
    await message.answer(
        "Что вам интересно?",
        reply_markup=builder.as_markup()
    )

@router.message(F.text.lower() == "код")
async def codeChoice(message: types.Message, state: FSMContext):
    await state.set_state(None)
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="C#",
        callback_data="c#")
    )
    builder.add(types.InlineKeyboardButton(
        text="Vue.js",
        callback_data="js")
    )
    await message.answer(
        "Какую технологию использовать?",
        reply_markup=builder.as_markup()
    )

@router.message(F.text.lower() == "аналитика")
async def codeChoice(message: types.Message, state: FSMContext):
    await state.set_state(None)
    builder = InlineKeyboardBuilder()
    builder.row(types.InlineKeyboardButton(
        text="Написать user-cases",
        callback_data="analisis")
    )
    builder.row(types.InlineKeyboardButton(
        text="Подготовить план презентации",
        callback_data="presentation")
    )
    builder.row(types.InlineKeyboardButton(
        text="Распределить задачи",
        callback_data="tasks")
    )
    await message.answer(
        "Что необходимо сделать?",
        reply_markup=builder.as_markup()
    )

@router.message(F.text.lower() == "дизайн")
async def design(message: types.Message, state: FSMContext):
    await state.set_state(None)
    builder = InlineKeyboardBuilder()
    builder.row(types.InlineKeyboardButton(
        text="Создать логотип",
        callback_data="logo")
    )
    await message.answer(
        "Что необходимо сделать?",
        reply_markup=builder.as_markup()
    )

@router.message(Command("help"))
async def process_help_command(message: types.Message, state: FSMContext):
    await state.set_state(None)
    await message.answer("Это бот, помогающий команде на хакатоне. Выбери команду из меню.")

@router.message(Task_descr.opisaniye)
async def descr_chosen(message: types.Message, state: FSMContext):
    await state.update_data(zadaniye_descr=message.text)
    data = await state.get_data()
    await state.set_state(None)
    if (('last_call' in data) and (data["last_call"] != None)):
        await message.answer(
        text="Описание сохранено. Запрос отправлен в модель"
        )
        match data["last_call"]:
            case "name_ideas":
                res = await name_ideas_Call(state)
                await message.answer(res)
            case "code":
                res = await code_Call(state)
                await message.answer(f"{res}", parse_mode=None)
            case "codeFront":
                res = await codeFront_Call(state)
                await message.answer(f"{res}", parse_mode=None)
            case "analisis":
                res = await analisis_Call(state)
                await message.answer(res)
            case "presentation":
                res = await presentation_Call(state)
                await message.answer(res)
            case "tasks":
                res = await tasks_utils(state)
                await message.answer(res)
            case "logo":
                image_url = await logo_Call(state)
                await message.answer_photo(image_url)
    else:
        await message.answer(
        text="Описание сохранено"
    )