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
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
)
# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
        ("human", "{input}")
    ]
)

chain1 = prompt | llm

@router.message(Command("llm"))
async def langchainllm(message: types.Message, command: CommandObject):
    response = chain1.invoke({"input": command.args})
    await message.answer(response.content)


def get_docs():
    loader = WebBaseLoader('https://python.langchain.com/docs/expression_language/')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
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
    Answer the user's question.
    Context: {context}
    Question: {input}
    """)

    # chain = prompt | model
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt2
    )

    retriever = vectorStore.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

#docs = get_docs()
#vectorStore = create_vector_store(docs)
#chain2 = create_chain(vectorStore)

#@router.message(Command("llm2"))
#async def langchainllm(message: types.Message):
    #response = chain2.invoke({"input": "What is LCEL?"})
    #await message.answer(response["answer"])

@router.message(Command("help"))
async def process_help_command(message: types.Message):
    await message.answer("Напиши мне что-нибудь, и я отпрпавлю этот текст тебе в ответ!")

@router.callback_query(F.data == "ideas")
async def ideas(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты креативный менеджер в компьютерной компании"},
            {"role": "user", "content": "Придумай 10 идей для программы в сфере городской инфраструктуры"},
        ]
    )
    res = response.choices[0].message.content

    await callback.message.answer(res)
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

class Task_descr(StatesGroup):
    opisaniye = State()
    name = State()

@router.callback_query(F.data == "name_ideas")
async def name_ideas(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = await name_ideas_Call(state)
        await state.update_data(last_call=None)
        await callback.message.answer(res)
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="name_ideas")
        await callback.message.answer(res)
    await callback.answer()

async def code_Call(state: FSMContext):
    chat_history = []
    data = await state.get_data()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты программист, пишущий на языке C# c использованием фреймворка ASP.Net Core"},
            {"role": "user", "content": f"Напиши API для сервиса. Описание проекта: {data['zadaniye_descr']}. Предоставь код программы полностью"},
        ]
    )
    chat_history.append(HumanMessage(content=f"Напиши API для сервиса. Описание проекта: {data['zadaniye_descr']}. Предоставь код программы полностью"))
    chat_history.append(AIMessage(content=response.choices[0].message.content))
    await state.update_data(last_call=None)
    return response.choices[0].message.content

async def code_Call_history(state: FSMContext):
    chat_history = []
    data = await state.get_data()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты программист, пишущий на языке C# c использованием фреймворка ASP.Net Core"},
            MessagesPlaceholder(variable_name="chat_history"),
            {"role": "user", "content": f"Напиши API для сервиса. Описание проекта: {data['zadaniye_descr']}. Предоставь код программы полностью"},
        ]
    )
    chat_history.append(HumanMessage(content=f"Напиши API для сервиса. Описание проекта: {data['zadaniye_descr']}. Предоставь код программы полностью"))
    chat_history.append(AIMessage(content=response.choices[0].message.content))
    await state.update_data(last_call=None)
    return response.choices[0].message.content

@router.callback_query(F.data == "c#")
async def code(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = code_Call(state)
        await callback.message.answer(f"{res}", parse_mode=None)
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="code")
        await callback.message.answer(res)
    await callback.answer()

async def codeFront_Call(state: FSMContext):
    data = await state.get_data()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты фронтенд-разработчик, пишущий на языке js c использованием фреймворка Vue.js, а также дизайнер-верстальщик"},
            {"role": "user", "content": f"Напиши фронтеннд с оригинальным дизайном для сервиса Помощника по организации стажировок и практик. Описание проекта: {data['zadaniye_descr']}"},
        ]
    )
    await state.update_data(last_call=None)
    return response.choices[0].message.content

@router.callback_query(F.data == "js")
async def codeFront(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = codeFront_Call(state)
        await callback.message.answer(f"{res}", parse_mode=None)
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="codeFront")
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
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = analisis_Call(state)
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание"
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
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = presentation_Call(state)
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="presentation")
    await callback.message.answer(res)
    await callback.answer()

async def tasks_Call(state: FSMContext):
    data = await state.get_data()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты аналитик в команде программистов на хакатоне"},
            {"role": "user", "content": f"Напиши список задач, которые команде необходимо реализовать для подготовки проекта. Распредели эти задачи между 4 учасниками команды: бэкенд-разработчик, фронтенд-разработчик, дизайнер, аналитик. Описание проекта: {data['zadaniye_descr']}"},
        ]
    )
    await state.update_data(last_call=None)
    return response.choices[0].message.content

@router.callback_query(F.data == "tasks")
async def tasks(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        res = tasks_Call(state)
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание"
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
    data = await state.get_data()
    if ('zadaniye_descr' in data):
        image_url = logo_Call(state)
        await callback.message.reply_photo(image_url)
    else:
        res = f"В настоящий момент задание на хакатон не указано. \n \nУкажите задание"
        await state.set_state(Task_descr.opisaniye)
        await state.update_data(last_call="logo")
        await callback.message.answer(res)

    await callback.answer()