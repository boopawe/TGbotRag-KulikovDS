import logging
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import BotCommand, Message
from dotenv import load_dotenv
import os
from database import KnowledgeBase
from llm import ModelInterface

load_dotenv()

TOKEN = os.getenv("BOT_KEY")
knowledge_base = KnowledgeBase()
ai_model = ModelInterface()
bot = Bot(token=TOKEN)
dp = Dispatcher()

async def register_commands(bot: Bot):
    commands_list = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="add", description="Добавить факт: /add <текст>"),
        BotCommand(command="get_all", description="Показать все факты"),
        BotCommand(command="generate", description="Спросить ИИ: /generate <текст>"),
        BotCommand(command="rag", description="Ответ с поиском: /rag <текст>"),
        BotCommand(command="clear_db", description="Очистить базу данных"),
        BotCommand(command="help", description="Список команд")
    ]
    await bot.set_my_commands(commands_list)

@dp.message(Command("help"))
async def show_help(message: Message):
    help_text = (
        "Доступные команды:\n\n"
        "/add <текст> - добавить факт в базу знаний\n"
        "/get_all - показать все сохранённые факты\n"
        "/generate <вопрос> - прямой ответ ИИ без поиска в базе\n"
        "/rag <вопрос> - поиск в базе и ответ на основе найденных фактов\n"
        "/clear_db - полностью очистить базу данных\n"
        "/help - показать это сообщение"
    )
    await message.reply(help_text)

@dp.message(Command("start"))
async def start_handler(message: Message):
    await message.answer(
        "Привет! Я бот с искусственным интеллектом.\n"
        "Могу отвечать на вопросы и запоминать факты.\n"
        "Используй /rag для поиска по базе знаний.\n"
        "Введи /help для списка всех команд."
    )

@dp.message(Command("add"))
async def add_fact(message: Message):
    fact = message.text.replace("/add", "").strip()
    if not fact:
        await message.reply("Пример: /add Земля круглая")
        return
    knowledge_base.store_information(fact)
    await message.reply("Факт успешно добавлен в базу знаний!")

@dp.message(Command("get_all"))
async def get_all_facts(message: Message):
    facts = knowledge_base.get_all_records()
    if not facts.strip():
        await message.reply("База знаний пока пуста.")
    else:
        await message.reply(f"Моя база знаний:\n\n{facts}")

@dp.message(Command("generate"))
async def generate(message: Message):
    prompt = message.text.replace("/generate", "").strip()
    if not prompt:
        await message.reply("Пример: /generate Почему небо голубое?")
        return
    
    status_msg = await message.reply("Думаю над ответом...")
    
    try:
        answer = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, ai_model.get_response, prompt),
            timeout=45.0
        )
        await status_msg.delete()
        await message.reply(f"Ответ:\n\n{answer}")
    except asyncio.TimeoutError:
        await status_msg.edit_text("Превышено время ожидания. Попробуйте позже или задайте более простой вопрос.")
    except Exception as e:
        await status_msg.edit_text(f"Ошибка: {str(e)}")

@dp.message(Command("rag"))
async def rag(message: Message):
    query = message.text.replace("/rag", "").strip()
    if not query:
        await message.reply("Пример: /rag Сколько лет вселенной?")
        return
    
    status_msg = await message.reply("Ищу информацию в базе знаний...")
    
    try:
        # Ищем до 3 похожих фактов
        context = knowledge_base.find_similar(query, results_count=2)
        
        if not context or not context.strip():
            context = "Информация по вашему запросу отсутствует в базе знаний."
            await status_msg.edit_text("База не содержит точных данных. Отвечаю без контекста...")
        else:
            await status_msg.edit_text(f"Найдены факты в базе.\n\nГенерирую развёрнутый ответ...")
        
        # Улучшенный промпт для развёрнутого ответа на основе базы
        prompt = f"""На основе следующей информации из базы знаний ответь на вопрос пользователя. 
Дай подробный, развёрнутый ответ. Объясни основные моменты.

Информация из базы знаний:
{context}

Вопрос пользователя: {query}

Развёрнутый ответ:"""
        
        answer = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, ai_model.get_response, prompt),
            timeout=45.0
        )
        
        await status_msg.delete()
        await message.reply(f"Ответ на основе базы знаний:\n\n{answer}")
        
    except asyncio.TimeoutError:
        await status_msg.edit_text("Превышено время ожидания. Попробуйте позже.")
    except Exception as e:
        await status_msg.edit_text(f"Ошибка: {str(e)}")

@dp.message(Command("clear_db"))
async def clear_db(message: Message):
    knowledge_base.clear_all_records()
    await message.reply("База данных полностью очищена!")

async def launch_bot():
    await bot.delete_webhook(drop_pending_updates=True)
    await register_commands(bot)
    print("Бот успешно запущен и готов к работе!")
    print("Модель TinyLlama-1.1B загружена")
    print("База данных готова")
    print("\nДоступные команды:")
    print("/add, /get_all, /generate, /rag, /clear_db, /help")
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(launch_bot())