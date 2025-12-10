# ============================================================
# ------------------------------------------------------------
# ВНИМАНИЕ: если вы запускаете ноутбук повторно и пакеты уже
# установлены, эту ячейку можно закомментировать или пропустить.
# ============================================================

!pip install -q openai \
                requests \
                beautifulsoup4 \
                pandas \
                "scikit-learn>=1.0.0"

# ============================================================
# ------------------------------------------------------------
# Здесь ничего "умного" не происходит, просто собираем все
# нужные зависимости в одном месте.
# ============================================================

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import requests  # для HTTP-запросов (Google Custom Search и загрузка HTML)
from bs4 import BeautifulSoup  # для парсинга HTML страниц

import pandas as pd  # для табличного представления корпуса и результатов

from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF векторизация
from sklearn.metrics.pairwise import cosine_similarity       # косинусное сходство

from openai import OpenAI  # клиент для Proxy API ChatGPT

# ============================================================
# В реальных проектах ключи нужно хранить в переменных окружения
# или в .env-файле. Здесь, по условию задания, используются
# тестовые ключи, которые допускается держать в коде.
# ============================================================

# --- Proxy API ChatGPT (OpenAI-совместимый клиент) ---

# Можно переопределить через переменные окружения, если нужно
PROXYAPI_KEY = os.getenv(
    "PROXYAPI_KEY",
    "sk-Y2VSk9ZKuCJbQD9xO3jp0jVxlJsGynOz"  # тестовый ключ
)

PROXYAPI_BASE_URL = os.getenv(
    "PROXYAPI_BASE_URL",
    "https://openai.api.proxyapi.ru/v1"
)

# Инициализация клиента OpenAI через Proxy API
llm_client = OpenAI(
    api_key=PROXYAPI_KEY,
    base_url=PROXYAPI_BASE_URL,
)

# Модель по умолчанию (при необходимости можно поменять)
# Например: "anthropic/claude-sonnet-4-20250514" или другая,
# поддерживаемая Proxy API.
DEFAULT_LLM_MODEL = os.getenv(
    "DEFAULT_LLM_MODEL",
    "anthropic/claude-sonnet-4-20250514"
)


# --- Google Custom Search API (для поиска примерных ТЗ) ---

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY",
    "AIzaSyDfSS4_mrxnYunncB8jNrEYhzdG6xKNnVo"  # тестовый ключ из задания
)

SEARCH_ENGINE_ID = os.getenv(
    "SEARCH_ENGINE_ID",
    "0483b4fa074fb4926"  # идентификатор поискового движка из задания
)

print("Проверка настроек:")
print(" - PROXYAPI_BASE_URL:", PROXYAPI_BASE_URL)
print(" - DEFAULT_LLM_MODEL:", DEFAULT_LLM_MODEL)
print(" - GOOGLE_API_KEY установлен:", bool(GOOGLE_API_KEY))
print(" - SEARCH_ENGINE_ID:", SEARCH_ENGINE_ID)
