from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List, Union, Any, Literal, Tuple
import logging
import os
import shutil
import google.generativeai as genai
from dotenv import load_dotenv
import json
from datetime import datetime
import base64
from PIL import Image
import io
import redis
import pickle
import re
from redis.connection import ConnectionPool
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Redis configuration
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': 0,
    'decode_responses': False,
    'max_connections': 10,
    'socket_timeout': 5,
    'socket_connect_timeout': 5,
    'retry_on_timeout': True
}

# Create Redis connection pool
redis_pool = ConnectionPool(**REDIS_CONFIG)

def get_redis_client():
    """Create a new Redis client from the connection pool"""
    try:
        client = redis.Redis(connection_pool=redis_pool)
        client.ping()
        logger.info("Successfully connected to Redis")
        return client
    except redis.ConnectionError as e:
        logger.warning(f"Could not connect to Redis: {str(e)}")
        return None

# Initialize Redis
redis_client = get_redis_client()
if not redis_client:
    logger.warning("Running without Redis - some features may be limited")

# Конфигурация приложения
app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv('ALLOWED_ORIGINS', '*')],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Создание директории для загрузок
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Конфигурация Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY не найден в переменных окружения")

# Load datasets
def load_dataset(filename: str) -> Dict:
    """Load and validate a dataset file"""
    try:
        file_path = Path("data") / filename
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {filename}")
        return data
    except Exception as e:
        logger.error(f"Error loading {filename}: {str(e)}")
        return {}

# Initialize datasets
try:
    products_dataset = load_dataset('products_dataset_complete.json')
    se_dataset = load_dataset('se_dataset.json')
    skin_type_questionnaire = load_dataset('skin_type_questionnaire.json')
    logger.info("All datasets loaded successfully")
except Exception as e:
    logger.error(f"Error loading datasets: {str(e)}")
    products_dataset = {}
    se_dataset = {}
    skin_type_questionnaire = {}

# Initialize Gemini model
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Gemini model: {str(e)}")
    model = None

# Initialize embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Embedding model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing embedding model: {str(e)}")
    embedding_model = None

# Модели данных
class ChatRequest(BaseModel):
    message: str
    session_id: str
    image_data: Optional[str] = None

class SessionAnalysis(BaseModel):
    session_id: str

class SkinType(str):
    DRY = "dry"
    OILY = "oily"
    COMBINATION = "combination"
    SENSITIVE = "sensitive"

class Category(str):
    ACNE = "acne"
    OILY_SKIN = "oily_skin"
    HYDRATION = "hydration"
    ANTI_AGING = "anti_aging"

class RoutineStep(str):
    CLEANSING = "cleansing"
    TONING = "toning"
    MOISTURIZING = "moisturizing"
    SUN_PROTECTION = "sun_protection"
    EXFOLIANT = "exfoliant"
    SERUM = "serum"
    MASK = "mask"
    SPECIAL_TREATMENT = "special_treatment"

class DialogContext:
    def __init__(self):
        self.skin_type: Optional[str] = None
        self.category: Optional[str] = None
        self.concerns: List[str] = []
        self.questionnaire_step: int = 0
        self.last_recommendations = None
        self.confidence_threshold = 0.7

    def update_from_message(self, message: str) -> None:
        """Update context based on message content"""
        message = message.lower()
        
        # Extract skin type if mentioned
        skin_types = {
            'normal': ['normal', 'нормальная', 'нормальной'],
            'oily': ['oily', 'жирная', 'жирной'],
            'dry': ['dry', 'сухая', 'сухой'],
            'combination': ['combination', 'комбинированная', 'комбинированной'],
            'sensitive': ['sensitive', 'чувствительная', 'чувствительной']
        }
        
        for skin_type, keywords in skin_types.items():
            if any(keyword in message for keyword in keywords):
                self.skin_type = skin_type
                logger.info(f"Updated skin type to: {skin_type}")
                break

        # Extract category if mentioned
        categories = {
            'acne': ['acne', 'акне', 'прыщи', 'угри'],
            'oily_skin': ['oily skin', 'жирная кожа', 'жирность'],
            'hydration': ['hydration', 'увлажнение', 'сухость'],
            'anti_aging': ['anti-aging', 'anti aging', 'морщины', 'старение']
        }
        
        for category, keywords in categories.items():
            if any(keyword in message for keyword in keywords):
                self.category = category
                logger.info(f"Updated category to: {category}")
                break

        # Extract concerns
        concern_keywords = {
            'acne': ['acne', 'акне', 'прыщи', 'угри'],
            'aging': ['aging', 'anti-aging', 'wrinkles', 'старение', 'морщины'],
            'pigmentation': ['pigmentation', 'dark spots', 'пигментация', 'пятна'],
            'dryness': ['dryness', 'сухость'],
            'sensitivity': ['sensitivity', 'чувствительность'],
            'redness': ['redness', 'покраснение', 'краснота'],
            'pores': ['pores', 'поры'],
            'blackheads': ['blackheads', 'черные точки'],
            'oiliness': ['oiliness', 'жирность'],
            'dullness': ['dullness', 'тусклость'],
            'uneven tone': ['uneven tone', 'неровный тон'],
            'dehydration': ['dehydration', 'обезвоживание']
        }
        
        for concern, keywords in concern_keywords.items():
            if any(keyword in message for keyword in keywords):
                if concern not in self.concerns:
                    self.concerns.append(concern)
                    logger.info(f"Added concern: {concern}")

    def get_missing_info(self) -> List[str]:
        """Return what information is still needed"""
        missing = []
        if not self.skin_type:
            missing.append("skin type")
        if not self.category and not self.concerns:
            missing.append("skin concerns or category")
        return missing

# Вспомогательные функции
def sanitize_filename(filename: str) -> str:
    """Санитизация имени файла"""
    # Удаляем небезопасные символы
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    # Ограничиваем длину
    return filename[:100]

def validate_image_format(file: UploadFile) -> bool:
    """Проверка формата изображения"""
    allowed_formats = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
    return file.content_type in allowed_formats

def process_image_data(image_data: str) -> tuple:
    """Обработка и валидация данных изображения"""
    try:
        # Проверяем размер base64 строки (примерно 4/3 от размера файла)
        if len(image_data) > 5.3 * 1024 * 1024:  # Учитываем base64 overhead
            raise ValueError("Изображение слишком большое. Максимальный размер - 4MB")
        
        # Декодируем base64
        image_bytes = base64.b64decode(image_data)
        
        # Определяем формат изображения
        image = Image.open(io.BytesIO(image_bytes))
        format = image.format.lower()
        if format not in {'jpeg', 'png', 'gif', 'webp'}:
            raise ValueError(f"Неподдерживаемый формат изображения: {format}")
        
        mime_type = f"image/{format}"
        return image_bytes, mime_type
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Ошибка при обработке изображения: {str(e)}")

def get_session_history(session_id: str) -> List[dict]:
    """Get session history from Redis or memory"""
    try:
        if redis_client:
            data = redis_client.get(f"session:{session_id}:history")
            if data:
                return pickle.loads(data)
        return []
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        return []

def save_session_history(session_id: str, history: List[dict]):
    """Save session history to Redis or memory"""
    try:
        if redis_client:
            redis_client.setex(
                f"session:{session_id}:history",
                3600 * 24 * 7,  # Store for 7 days
                pickle.dumps(history)
            )
    except Exception as e:
        logger.error(f"Error saving session history: {str(e)}")

def get_session_metadata(session_id: str) -> dict:
    """Получает метаданные сессии из Redis"""
    try:
        data = redis_client.get(f"session:{session_id}:metadata")
        if data:
            return pickle.loads(data)
        return {"last_activity": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Ошибка при получении метаданных сессии: {str(e)}")
        return {"last_activity": datetime.now().isoformat()}

def save_session_metadata(session_id: str, metadata: dict):
    """Сохраняет метаданные сессии в Redis"""
    try:
        redis_client.setex(
            f"session:{session_id}:metadata",
            3600 * 24 * 7,  # Храним 7 дней
            pickle.dumps(metadata)
        )
    except Exception as e:
        logger.error(f"Ошибка при сохранении метаданных сессии: {str(e)}")

def format_ai_response(text: str) -> str:
    """Форматирует ответ AI в HTML с безопасной обработкой"""
    # Разделяем текст на параграфы
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []
    
    # Паттерны для выделения важных компонентов
    patterns = [
        (r'(\d+(?:\.\d+)?%)', r'<span class="concentration">\1</span>'),  # Концентрации
        (r'\b(гиалуроновая кислота|ретинол|витамин C|ниацинамид|салициловая кислота|AHA|BHA|пептиды|керамиды)\b', 
         r'<span class="ingredient">\1</span>'),  # Ингредиенты
        (r'(Внимание|Предупреждение|Важно):', r'<span class="warning">\1:</span>'),  # Предупреждения
        (r'(увлажняет|успокаивает|восстанавливает|защищает|укрепляет|осветляет|омолаживает|очищает)\b',
         r'<span class="benefit">\1</span>')  # Преимущества
    ]
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        # Удаляем звездочки из заголовков
        p = p.replace('**', '')
        
        # Применяем выделение к тексту
        for pattern, replacement in patterns:
            p = re.sub(pattern, replacement, p, flags=re.IGNORECASE)
            
        # Обработка списков
        if p.startswith('1.') or p.startswith('-'):
            items = p.split('\n')
            formatted_items = []
            for item in items:
                item = item.strip()
                if item:
                    # Удаляем маркеры списка
                    item = re.sub(r'^[0-9]+\.\s*|^-\s*', '', item)
                    # Выделяем названия продуктов
                    if ':' in item:
                        product_name, description = item.split(':', 1)
                        item = f'<span class="product-name">{product_name}</span>:{description}'
                    formatted_items.append(f'<li>{item}</li>')
            if formatted_items:
                formatted_paragraphs.append(f'<ul class="skincare-list">{"".join(formatted_items)}</ul>')
        # Обработка заголовков
        elif p.startswith('SKIN ANALYSIS') or p.startswith('PRODUCT RECOMMENDATIONS') or \
             p.startswith('SCIENTIFIC EVIDENCE') or p.startswith('CARE ROUTINE') or \
             p.startswith('FOLLOW-UP QUESTIONS'):
            formatted_paragraphs.append(f'<h3 class="section-header">{p}</h3>')
        # Обработка подзаголовков (Morning/Evening)
        elif p.lower().startswith('morning') or p.lower().startswith('evening'):
            formatted_paragraphs.append(f'<h4 class="subsection-header">{p}</h4>')
        # Обработка обычных параграфов
        else:
            formatted_paragraphs.append(f'<p class="skincare-paragraph">{p}</p>')
    
    return ''.join(formatted_paragraphs)

# Глобальные переменные для хранения датасетов и их эмбеддингов
products_data = None
research_data = None
products_embeddings = None
research_embeddings = None

def load_datasets():
    """Загрузка и предобработка датасетов"""
    global products_data, research_data, products_embeddings, research_embeddings
    
    try:
        # Загрузка датасета продуктов
        with open('data/products_dataset_complete.json', 'r', encoding='utf-8') as f:
            products_data = json.load(f)
        
        # Загрузка научного датасета
        with open('data/se_dataset.json', 'r', encoding='utf-8') as f:
            research_data = json.load(f)
        
        # Создание текстовых описаний для продуктов
        product_texts = []
        for product in products_data:
            text = f"{product.get('name', '')} {product.get('description', '')} "
            text += f"{product.get('ingredients', '')} {product.get('benefits', '')}"
            product_texts.append(text)
        
        # Создание текстовых описаний для исследований
        research_texts = []
        for article in research_data:
            text = f"{article.get('title', '')} {article.get('abstract', '')} "
            text += f"{article.get('conclusion', '')}"
            research_texts.append(text)
        
        # Создание эмбеддингов
        products_embeddings = embedding_model.encode(product_texts)
        research_embeddings = embedding_model.encode(research_texts)
        
        logging.info("Датасеты успешно загружены и обработаны")
        
    except Exception as e:
        logging.error(f"Ошибка при загрузке датасетов: {str(e)}")
        raise

def find_relevant_products(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Поиск релевантных продуктов по запросу"""
    if products_embeddings is None:
        load_datasets()
    
    query_embedding = embedding_model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], products_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [products_data[i] for i in top_indices]

def find_relevant_research(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Поиск релевантных исследований по запросу"""
    if research_embeddings is None:
        load_datasets()
    
    query_embedding = embedding_model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], research_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [research_data[i] for i in top_indices]

def generate_recommendations_prompt(query: str, products: List[dict], research: List[dict]) -> str:
    """Генерация промпта для Gemini на основе найденной информации"""
    prompt = f"""На основе следующего запроса пользователя и предоставленной информации, дайте персонализированные рекомендации по уходу за кожей:

Запрос пользователя: {query}

Релевантные продукты:
{json.dumps(products, indent=2, ensure_ascii=False)}

Релевантные исследования:
{json.dumps(research, indent=2, ensure_ascii=False)}

Пожалуйста, предоставьте:
1. Анализ потребностей пользователя
2. Рекомендации по продуктам с объяснением их эффективности
3. Научное обоснование рекомендаций
4. Предупреждения и противопоказания, если есть
5. Общий план ухода

Ответ должен быть структурированным, научно обоснованным и понятным для пользователя."""
    
    return prompt

def store_message(role: str, content: str):
    """Сохранение сообщения в истории"""
    try:
        # Получение текущей истории
        history = get_session_history()
        
        # Добавление нового сообщения
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Ограничение истории последними 100 сообщениями
        history = history[-100:]
        
        # Сохранение в Redis
        save_session_history(history)
        
    except Exception as e:
        logging.error(f"Ошибка при сохранении сообщения: {str(e)}")

def get_session_history() -> List[dict]:
    """Получение истории сообщений из Redis"""
    try:
        data = redis_client.get("chat_history")
        if data:
            return pickle.loads(data)
        return []
    except Exception as e:
        logging.error(f"Ошибка при получении истории: {str(e)}")
        return []

def save_session_history(history: List[dict]):
    """Сохранение истории сообщений в Redis"""
    try:
        redis_client.setex(
            "chat_history",
            3600 * 24 * 7,  # Храним 7 дней
            pickle.dumps(history)
        )
    except Exception as e:
        logging.error(f"Ошибка при сохранении истории: {str(e)}")

# Skin type categories
SKIN_TYPES = {
    "acne": "Acne-prone skin",
    "oily": "Oily skin",
    "dry": "Dry skin",
    "combination": "Combination skin",
    "normal": "Normal skin",
    "sensitive": "Sensitive skin",
    "mature": "Mature skin"
}

def detect_skin_type(message: str) -> str:
    """Detect skin type from user message"""
    message = message.lower()
    for key, value in SKIN_TYPES.items():
        if key in message or value.lower() in message:
            return value
    return None

def generate_skin_type_prompt() -> str:
    """Generate prompt for skin type clarification"""
    return """Please specify your skin type from the following categories:
1. Acne-prone skin
2. Oily skin
3. Dry skin
4. Combination skin
5. Normal skin
6. Sensitive skin
7. Mature skin

You can either select a number or type the skin type directly."""

def generate_structured_response(query: str, skin_type: str, products: List[dict], research: List[dict]) -> dict:
    """Generate structured response with recommendations"""
    return {
        "skin_type": skin_type,
        "analysis": {
            "user_query": query,
            "skin_concerns": [],  # To be filled by Gemini
            "recommendations": []  # To be filled by Gemini
        },
        "products": products,
        "research": research,
        "care_plan": {
            "morning": [],  # To be filled by Gemini
            "evening": [],  # To be filled by Gemini
            "weekly": []    # To be filled by Gemini
        },
        "search_links": {
            "products": f"https://example.com/search?q={skin_type}+skincare+products",
            "research": f"https://example.com/search?q={skin_type}+skincare+research"
        }
    }

# Эндпоинты
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.info("Запрос главной страницы")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "").strip()
        session_id = data.get("session_id")

        if not message:
            return JSONResponse(
                status_code=400,
                content={"response": "Message cannot be empty"}
            )

        if not model:
            return JSONResponse(
                status_code=500,
                content={"response": "Service temporarily unavailable. Please try again later."}
            )

        # Detect user language
        try:
            user_language = detect(message)
        except:
            user_language = "en"  # Default to English if detection fails

        # Get or create context
        context = dialog_contexts.get(session_id)
        if not context:
            context = DialogContext()
            dialog_contexts[session_id] = context

        # Update context with message
        context.update_from_message(message)
        
        try:
            # Find relevant products and research
            relevant_products = []
            relevant_research = []
            
            if embedding_model:
                try:
                    relevant_products = find_relevant_products(message)
                    relevant_research = find_relevant_research(message)
                except Exception as e:
                    logger.error(f"Error finding relevant data: {str(e)}")

            # Generate response using Gemini
            prompt = f"""You are a skincare expert. The user message is: {message}

Based on our product database and scientific research, provide a structured response in {user_language} with the following sections:

SKIN ANALYSIS
Analyze the user's skin type and concerns based on their message.
Reference specific products from our database that match their needs.

PRODUCT RECOMMENDATIONS
List 2-3 most suitable products from our database.
For each product, explain why it's recommended based on its ingredients and benefits.
Include product names, brands, and key ingredients.

SCIENTIFIC EVIDENCE
Reference relevant research from our database to support your recommendations.
Explain the scientific basis for the suggested products and ingredients.

CARE ROUTINE
Provide a simple daily routine using the recommended products.
Include morning and evening steps.

If the user hasn't provided enough information, structure your questions like this:

FOLLOW-UP QUESTIONS
Ask about their skin type.
Ask about specific concerns they want to address.

Keep the response in {user_language} and maintain this structured format with headers and separate paragraphs.
Each section should be clearly separated with double line breaks."""

            gemini_response = model.generate_content(prompt)
            
            if not gemini_response or not gemini_response.text:
                logger.error("Empty response from Gemini API")
                return JSONResponse(
                    content={"response": "I apologize, but I couldn't generate a response. Please try again."}
                )

            # Format response using format_ai_response
            raw = gemini_response.text.strip()
            html = format_ai_response(raw)

            return JSONResponse(
                content={"response": html}
            )

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return JSONResponse(
                content={"response": "I apologize, but there was an error processing your request. Please try again."}
            )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"response": "An error occurred while processing your request."}
        )

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Получен запрос на загрузку изображения: {file.filename}")
        
        # Проверка типа файла
        if not validate_image_format(file):
            logger.error("Неподдерживаемый формат изображения")
            raise HTTPException(status_code=400, detail="Неподдерживаемый формат изображения")
        
        # Проверка размера файла
        if file.size > 4 * 1024 * 1024:
            logger.error("Размер файла превышает 4MB")
            raise HTTPException(status_code=400, detail="Размер файла не должен превышать 4MB")
        
        # Санитизация имени файла
        safe_filename = sanitize_filename(file.filename)
        unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_filename}"
        file_location = os.path.join("static/uploads", unique_filename)
        
        # Сохранение файла
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        logger.info(f"Изображение успешно загружено: {unique_filename}")
        return {
            "filename": unique_filename,
            "message": "Фотография успешно загружена.",
            "url": f"/static/uploads/{unique_filename}"
        }
    except Exception as e:
        logger.error(f"Ошибка при загрузке изображения: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке файла: {str(e)}")

@app.get("/check-datasets")
async def check_datasets():
    """Проверка загрузки датасетов"""
    try:
        if products_data is None or research_data is None:
            load_datasets()
        
        return {
            "products_count": len(products_data),
            "research_count": len(research_data),
            "products_sample": products_data[0] if products_data else None,
            "research_sample": research_data[0] if research_data else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-rag")
async def test_rag(request: Request):
    """Тестирование RAG системы"""
    try:
        data = await request.json()
        query = data.get("query", "").strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Запрос не может быть пустым")
        
        # Поиск релевантной информации
        relevant_products = find_relevant_products(query)
        relevant_research = find_relevant_research(query)
        
        # Генерация промпта
        prompt = generate_recommendations_prompt(query, relevant_products, relevant_research)
        
        return {
            "query": query,
            "relevant_products": relevant_products,
            "relevant_research": relevant_research,
            "prompt": prompt
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Global storage for dialog contexts
dialog_contexts: Dict[str, DialogContext] = {}

def get_scientific_recommendations(category: str, skin_type: str) -> List[Dict]:
    """Retrieve relevant entries from se_dataset"""
    relevant_entries = []
    try:
        for entry in se_dataset:
            if (category.lower() in [tag.lower() for tag in entry.get('tags', [])] or
                skin_type.lower() in [tag.lower() for tag in entry.get('tags', [])]):
                relevant_entries.append({
                    'id': entry.get('id'),
                    'recommendation': entry.get('recommendation'),
                    'details': entry.get('details'),
                    'scientific_evidence': entry.get('evidence'),
                    'usage': entry.get('application_method'),
                    'frequency': entry.get('frequency'),
                    'reference': entry.get('link')
                })
        
        # Ensure at least 3 recommendations
        return relevant_entries[:max(3, len(relevant_entries))]
    except Exception as e:
        logger.error(f"Error getting scientific recommendations: {str(e)}")
        return []

def get_care_plan(category: str, skin_type: str) -> List[Dict]:
    """Generate care plan based on category and skin type"""
    basic_steps = [
        {
            "step": RoutineStep.CLEANSING,
            "description": "Gentle cleansing to remove impurities and prepare skin for treatment.",
            "scientific_reference": "Basic skincare principles, American Academy of Dermatology"
        },
        {
            "step": RoutineStep.TONING,
            "description": "Balance skin pH and prepare for treatments.",
            "scientific_reference": "Skin barrier function studies, 2023"
        },
        {
            "step": RoutineStep.MOISTURIZING,
            "description": "Hydrate and protect the skin barrier.",
            "scientific_reference": "Moisturizer efficacy studies, 2024"
        },
        {
            "step": RoutineStep.SUN_PROTECTION,
            "description": "Protect from UV damage and prevent premature aging.",
            "scientific_reference": "UV protection guidelines, WHO"
        }
    ]
    
    # Add category-specific steps
    if category == Category.ACNE:
        basic_steps.insert(2, {
            "step": RoutineStep.SPECIAL_TREATMENT,
            "description": "Target acne with appropriate active ingredients.",
            "scientific_reference": "Acne treatment guidelines, 2024"
        })
    elif category == Category.ANTI_AGING:
        basic_steps.insert(2, {
            "step": RoutineStep.SERUM,
            "description": "Apply anti-aging actives like retinol or peptides.",
            "scientific_reference": "Anti-aging interventions review, 2024"
        })
    
    return basic_steps

def get_product_recommendations(care_plan: List[Dict], skin_type: str, category: str) -> List[Dict]:
    """Find suitable products for each step in the care plan"""
    recommended_products = []
    try:
        for step in care_plan:
            step_products = []
            for product in products_dataset:
                if (product.get('category', '').lower() == step['step'].lower() and
                    (skin_type.lower() in [s.lower() for s in product.get('suitable_for', [])] or
                     'all skin types' in [s.lower() for s in product.get('suitable_for', [])])):
                    step_products.append({
                        'product_name': product.get('product_name'),
                        'ingredients': product.get('ingredients', []),
                        'purpose': product.get('purpose'),
                        'usage': product.get('instructions'),
                        'frequency': product.get('frequency'),
                        'link': product.get('link')
                    })
            
            # Take top 3 products for each step
            if step_products:
                recommended_products.extend(step_products[:3])
        
        return recommended_products
    except Exception as e:
        logger.error(f"Error getting product recommendations: {str(e)}")
        return []

def format_response(context: DialogContext, care_plan: List[Dict], 
                   scientific_recommendations: List[Dict], 
                   product_recommendations: List[Dict]) -> Dict:
    """Format the final response according to the specified structure"""
    response = {
        "response_type": "recommendation",
        "category": context.category,
        "skin_type": context.skin_type,
        "care_plan": care_plan,
        "scientific_recommendations": scientific_recommendations,
        "product_recommendations": product_recommendations,
        "search_suggestion": {
            "text": "Search online for best prices on these products",
            "url_template": "https://www.google.com/search?q={product_name}"
        }
    }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)