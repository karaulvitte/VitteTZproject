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
