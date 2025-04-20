import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Получение API ключа
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
    exit(1)

logger.info("Initializing Gemini with API key")
genai.configure(api_key=api_key)

# Создание модели
model = genai.GenerativeModel('gemini-pro')

try:
    logger.info("Sending test request to Gemini API")
    response = model.generate_content("Привет! Это тестовое сообщение. Ответь, пожалуйста, коротко.")
    logger.info("Response received successfully")
    logger.info(f"Response text: {response.text}")
except Exception as e:
    logger.error(f"Error during API call: {str(e)}", exc_info=True) 