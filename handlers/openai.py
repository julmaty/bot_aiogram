from aiogram import Bot, Router, types, F
from aiogram.filters.command import Command, CommandObject
from aiogram.client.default import DefaultBotProperties
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import State, StatesGroup
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from config_reader import config
import os

from openai import OpenAI

os.environ["OPENAI_API_KEY"] = config.openai_key_personal.get_secret_value()
router = Router()
tokenAI=config.openai_key_personal.get_secret_value()
tokenPC=config.pinecone_key.get_secret_value()
client = OpenAI(
  api_key=tokenAI
)
pc = Pinecone(api_key=tokenPC)

def get_docs(input_link):
    loader = WebBaseLoader(input_link)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(
        temperature=0.4,
        model="gpt-3.5-turbo"
    )
    prompt2 = ChatPromptTemplate.from_template("""
    Ответь на вопрос исходя из контекста.
    Контекст: {context}
    Вопрос: {input}
    """)
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt2
    )
    retriever = vectorStore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

class Task_descr(StatesGroup):
    opisaniye = State()
    code_history = State()
    codeFront_history = State()
    codePy_history = State()
    ideas = State()
    tasks = State()
    documentation = State()
    documentation_new = State()

async def documentation_get(user_input, state: FSMContext):
    data = await state.get_data()
    chain = create_chain(data['vector'])
    response = chain.invoke({"input": f"{user_input}"})
    return response["answer"]


@router.callback_query(F.data == "ideas")
async def ideas(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(Task_descr.ideas)
    await callback.message.answer("Укажите сферу, в которой необходимо создать приложение:")
    await callback.answer()

async def name_ideas_Call(state: FSMContext):
    data = await state.get_data()
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты креативный менеджер в компьютерной компании"},
                {"role": "user", "content": f"Придумай 10 оригинальных названий для программы. Описание проекта: {data['zadaniye_descr']}"},
            ]
        )
    await state.update_data(last_call=None)
    return response.choices[0].message.content



@router.callback_query(F.data == "name_ideas")
async def name_ideas(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = await name_ideas_Call(state)
        await state.update_data(last_call=None)
        await callback.message.answer(res)
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание:"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="name_ideas")
        await callback.message.answer(res)
    await callback.answer()

async def code_Call(state: FSMContext):
    await state.update_data(chat_code_history = [])
    data = await state.get_data()
    llm = ChatOpenAI(
        model="gpt-3.5-turbo"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Ты программист, пишущий на языке C# c использованием фреймворка ASP.Net Core"),
            ("human", f"Напиши API для сервиса. Описание проекта: {data['zadaniye_descr']}. Предоставь код программы полностью")
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"input": data['zadaniye_descr']})
    data["chat_code_history"].append(HumanMessage(content=f"Напиши API для сервиса. Описание проекта: {data['zadaniye_descr']}. Предоставь код программы полностью"))
    data["chat_code_history"].append(AIMessage(content=response.content))
    await state.update_data(last_call=None)
    await state.set_state(Task_descr.code_history)
    return response.content

async def code_Call_history(state: FSMContext, user_input):
    data = await state.get_data()
    llm = ChatOpenAI(
        model="gpt-3.5-turbo"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Ты программист, пишущий на языке C# c использованием фреймворка ASP.Net Core"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )
    chain = prompt | llm
    response = chain.invoke({
        "input": user_input,
        "chat_history": data["chat_code_history"]
    })
    await state.set_state(Task_descr.code_history)
    data["chat_code_history"].append(HumanMessage(content=user_input))
    data["chat_code_history"].append(AIMessage(content=response.content))
    return response.content

@router.callback_query(F.data == "c#")
async def code(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        if (('chat_code_history' in data) and data['chat_code_history'] != []):
            res = await code_Call_history(state, "Выведи имеющийся код")
            await callback.message.answer(f"{res}", parse_mode=None)
            builder = InlineKeyboardBuilder()
            builder.row(types.InlineKeyboardButton(
                text="Начать новую генерацию",
                callback_data="code_new")
            )
            await callback.message.answer(
            "Чтобы продолжить беседу по имеющумуся коду, введите вопрос. \n \nЧтобы начать новую генерацию кода по текущему заданию, нажмите кнопку.",
            reply_markup=builder.as_markup()
            )
        else:
            res = await code_Call(state)
            await callback.message.answer(f"{res}", parse_mode=None)
            await callback.message.answer(
            text="Если хотите задать дополнительные вопросы по коду, введите их"
            )
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание:"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="code")
        await callback.message.answer(res)
    await callback.answer()

@router.callback_query(F.data == "code_new")
async def code_new(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = await code_Call(state)
        await callback.message.answer(f"{res}", parse_mode=None)
        await callback.message.answer(
        text="Если хотите задать дополнительные вопросы по коду, введите их. \n\nИли выбедите другую команду в меню."
        )
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание:"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="code")
        await callback.message.answer(res)
    await callback.answer()

async def codeFront_Call(state: FSMContext):
    await state.update_data(chat_codeFront_history = [])
    data = await state.get_data()
    llm = ChatOpenAI(
        model="gpt-3.5-turbo"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Ты фронтенд-разработчик, пишущий на языке js c использованием фреймворка Vue.js, а также дизайнер-верстальщик"),
            ("human", f"Напиши фронтеннд с оригинальным дизайном для сервиса. Описание проекта: {data['zadaniye_descr']}. Предоставь код полностью")
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"input": data['zadaniye_descr']})
    data["chat_codeFront_history"].append(HumanMessage(content=f"Напиши фронтеннд с оригинальным дизайном для сервиса. Описание проекта: {data['zadaniye_descr']}. Предоставь код полностью"))
    data["chat_codeFront_history"].append(AIMessage(content=response.content))
    await state.update_data(last_call=None)
    await state.set_state(Task_descr.codeFront_history)
    return response.content

async def codeFront_Call_history(state: FSMContext, user_input):
    data = await state.get_data()
    llm = ChatOpenAI(
        model="gpt-3.5-turbo"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Ты фронтенд-разработчик, пишущий на языке js c использованием фреймворка Vue.js, а также дизайнер-верстальщик"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )
    chain = prompt | llm
    response = chain.invoke({
        "input": user_input,
        "chat_history": data["chat_codeFront_history"]
    })
    await state.set_state(Task_descr.codeFront_history)
    data["chat_codeFront_history"].append(HumanMessage(content=user_input))
    data["chat_codeFront_history"].append(AIMessage(content=response.content))
    return response.content

@router.callback_query(F.data == "js")
async def codeFront(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        if (('chat_codeFront_history' in data) and data['chat_codeFront_history'] != []):
            res = await codeFront_Call_history(state, "Выведи имеющийся код")
            await callback.message.answer(f"{res}", parse_mode=None)
            builder = InlineKeyboardBuilder()
            builder.row(types.InlineKeyboardButton(
                text="Начать новую генерацию",
                callback_data="codeFront_new")
            )
            await callback.message.answer(
            "Чтобы продолжить беседу по имеющумуся коду, введите вопрос. \n \nЧтобы начать новую генерацию кода по текущему заданию, нажмите кнопку.",
            reply_markup=builder.as_markup()
            )
        else:
            res = await codeFront_Call(state)
            await callback.message.answer(f"{res}", parse_mode=None)
            await callback.message.answer(
            text="Если хотите задать дополнительные вопросы по коду, введите их"
            )
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание:"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="codeFront")
        await callback.message.answer(res)
    await callback.answer()

@router.callback_query(F.data == "codeFront_new")
async def codeFront_new(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = await codeFront_Call(state)
        await callback.message.answer(f"{res}", parse_mode=None)
        await callback.message.answer(
        text="Если хотите задать дополнительные вопросы по коду, введите их. \n\nИли выбедите другую команду в меню."
        )
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание:"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="codeFront")
        await callback.message.answer(res)
    await callback.answer()


async def codePy_Call(state: FSMContext):
    await state.update_data(chat_codePy_history = [])
    data = await state.get_data()
    llm = ChatOpenAI(
        model="gpt-3.5-turbo"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Ты разработчик, пишущий на языке python c использованием Fast API"),
            ("human", f"Напиши API сервиса. Описание проекта: {data['zadaniye_descr']}. Предоставь код полностью")
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"input": data['zadaniye_descr']})
    data["chat_codePy_history"].append(HumanMessage(content=f"Напиши API сервиса. Описание проекта: {data['zadaniye_descr']}. Предоставь код полностью"))
    data["chat_codePy_history"].append(AIMessage(content=response.content))
    await state.update_data(last_call=None)
    await state.set_state(Task_descr.codePy_history)
    return response.content

async def codePy_Call_history(state: FSMContext, user_input):
    data = await state.get_data()
    llm = ChatOpenAI(
        model="gpt-3.5-turbo"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Ты разработчик, пишущий на языке python c использованием Fast API"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )
    chain = prompt | llm
    response = chain.invoke({
        "input": user_input,
        "chat_history": data["chat_codePy_history"]
    })
    await state.set_state(Task_descr.codePy_history)
    data["chat_codePy_history"].append(HumanMessage(content=user_input))
    data["chat_codePy_history"].append(AIMessage(content=response.content))
    return response.content

@router.callback_query(F.data == "python")
async def codeFront(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        if (('chat_codePy_history' in data) and data['chat_codePy_history'] != []):
            res = await codePy_Call_history(state, "Выведи имеющийся код")
            await callback.message.answer(f"{res}", parse_mode=None)
            builder = InlineKeyboardBuilder()
            builder.row(types.InlineKeyboardButton(
                text="Начать новую генерацию",
                callback_data="codePy_new")
            )
            await callback.message.answer(
            "Чтобы продолжить беседу по имеющумуся коду, введите вопрос. \n \nЧтобы начать новую генерацию кода по текущему заданию, нажмите кнопку.",
            reply_markup=builder.as_markup()
            )
        else:
            res = await codePy_Call(state)
            await callback.message.answer(f"{res}", parse_mode=None)
            await callback.message.answer(
            text="Если хотите задать дополнительные вопросы по коду, введите их"
            )
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание:"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="codePy")
        await callback.message.answer(res)
    await callback.answer()

@router.callback_query(F.data == "codePy_new")
async def codePy_new(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = await codePy_Call(state)
        await callback.message.answer(f"{res}", parse_mode=None)
        await callback.message.answer(
        text="Если хотите задать дополнительные вопросы по коду, введите их. \n\nИли выбедите другую команду в меню."
        )
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание:"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="codePy")
        await callback.message.answer(res)
    await callback.answer()


async def analisis_Call(state: FSMContext):
    data = await state.get_data()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты аналитик в команде программистов"},
            {"role": "user", "content": f"Напиши юзер-кейсы для сервиса. Описание проекта: {data['zadaniye_descr']}"},
        ]
    )
    await state.update_data(last_call=None)
    return response.choices[0].message.content

@router.callback_query(F.data == "analisis")
async def analisis(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = await analisis_Call(state)
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание:"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="analisis")
    await callback.message.answer(res)
    await callback.answer()


async def presentation_Call(state: FSMContext):
    data = await state.get_data()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты креативный аналитик в команде программистов"},
            {"role": "user", "content": f"Напиши план презентации сервиса. План должен включать в себя задачи, которые решает сервис, его актуальность, перспективы развития сервия и другие пункты. Формулировки должны быть продающими, сформулированы креативно. Описание проекта: {data['zadaniye_descr']}"},
        ]
    )
    await state.update_data(last_call=None)
    return response.choices[0].message.content

@router.callback_query(F.data == "presentation")
async def presentation(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = await presentation_Call(state)
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание:"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="presentation")
    await callback.message.answer(res)
    await callback.answer()

async def tasks_Call(state: FSMContext, user_input):
    data = await state.get_data()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты аналитик в команде программистов на хакатоне"},
            {"role": "user", "content": f"Напиши список задач, которые команде необходимо реализовать для подготовки проекта. Распредели эти задачи между учасниками команды. Состав команды: {user_input}. Описание проекта: {data['zadaniye_descr']}"},
        ]
    )
    await state.update_data(last_call=None)
    return response.choices[0].message.content

async def tasks_utils(state: FSMContext):
    await state.set_state(Task_descr.tasks)
    return "Укажите, какой состав имеет ваша команда:"

@router.callback_query(F.data == "tasks")
async def tasks(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        await state.set_state(Task_descr.tasks)
        res = "Укажите, какой состав имеет ваша команда:"
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание:"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="tasks")
    await callback.message.answer(res)
    await callback.answer()

async def logo_Call(state: FSMContext):
    data = await state.get_data()
    response = client.images.generate(
        model="dall-e-3",
        prompt=f"логотип для сервиса. Описание проекта: {data['zadaniye_descr']}",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    await state.update_data(last_call=None)
    return response.data[0].url

@router.callback_query(F.data == "logo")
async def logo(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(last_call=None)
    await state.set_state(None)
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        image_url = await logo_Call(state)
        await callback.message.reply_photo(image_url)
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание:"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="logo")
        await callback.message.answer(res)

    await callback.answer()

@router.message(Task_descr.ideas)
async def ideas_ans(message: types.Message, state: FSMContext):
    await state.set_state(None)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты креативный менеджер в компьютерной компании"},
            {"role": "user", "content": f"Придумай 10 идей для программы в сфере {message.text}"},
        ]
    )
    res = response.choices[0].message.content

    await message.answer(res)

@router.message(Task_descr.tasks)
async def ideas_ans(message: types.Message, state: FSMContext):
    await state.set_state(None)
    res = await tasks_Call(state, message.text)
    await message.answer(res)

@router.message(Task_descr.code_history)
async def code_repeat(message: types.Message, state: FSMContext):
    res = await code_Call_history(state, message.text)
    await message.answer(f"{res}", parse_mode=None)
    await message.answer(
        text="Если хотите задать дополнительные вопросы по коду, введите их. \n \nИли выберите другую задачу в меню."
        )
    
@router.message(Task_descr.codeFront_history)
async def codeFront_repeat(message: types.Message, state: FSMContext):
    res = await codeFront_Call_history(state, message.text)
    await message.answer(f"{res}", parse_mode=None)
    await message.answer(
        text="Если хотите задать дополнительные вопросы по коду, введите их. \n \nИли выберите другую задачу в меню."
        )
    
@router.message(Task_descr.codePy_history)
async def codePy_repeat(message: types.Message, state: FSMContext):
    res = await codePy_Call_history(state, message.text)
    await message.answer(f"{res}", parse_mode=None)
    await message.answer(
        text="Если хотите задать дополнительные вопросы по коду, введите их. \n \nИли выберите другую задачу в меню."
        )
    
@router.message(Task_descr.documentation_new)
async def documentation_new(message: types.Message, state: FSMContext):
    await message.answer(
        text="Ожидайте"
        )
    await state.update_data(documentation=message.text)
    docs = get_docs(message.text)
    vectorStore = create_vector_store(docs)
    await state.update_data(vector=vectorStore)
    await state.set_state(Task_descr.documentation)
    await message.answer(
        text="Документация по ссылке загружена. \n \nВведите вопрос, на который хотите получить ответ:"
        )
    
@router.message(Task_descr.documentation)
async def documentation_new(message: types.Message, state: FSMContext):
    res = await documentation_get(message.text, state)
    print(res)
    await message.answer(str(res))
    await message.answer(
        text="Введите другой вопрос или воспользуйтесь меню, чтобы выбрать другую задачу."
        )
    
@router.message()
async def def_message(message: types.Message):
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
        text="Документация")
    )
    builder.row(types.KeyboardButton(
        text="Задание на хакатон")
    )
    await message.answer("Я не понимаю. Выбери команду из меню.", reply_markup=builder.as_markup(resize_keyboard=True))