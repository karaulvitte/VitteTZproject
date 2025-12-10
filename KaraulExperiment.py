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

# ============================================================
# Каждый кейс описывает будущую систему и её контекст.
# Эти описания будут подаваться модели при генерации разделов ТЗ.
# ============================================================

test_projects: List[Dict[str, Any]] = [
    {
        "id": "hr_muiv",
        "name": "Система учета сотрудников",
        "description": (
            "Разработка информационной системы учета сотрудников "
            "частного образовательного учреждения высшего образования "
            "«Московский университет имени С. Ю. Витте». "
            "Система должна хранить сведения о штатном и совместительском "
            "персонале, поддерживать поиск по должности и подразделению, "
            "формировать отчеты для кадровой службы и руководства."
        ),
        "domain": "учет кадров в вузе",
    },
    {
        "id": "vkr_support",
        "name": "Система поддержки ВКР",
        "description": (
            "Создание информационной системы поддержки подготовки и хранения "
            "выпускных квалификационных работ студентов. "
            "Система должна позволять регистрировать темы ВКР, закреплять "
            "научных руководителей, загружать промежуточные и итоговые версии "
            "работ, а также обеспечивать доступ к архиву ВКР с учетом ролей "
            "пользователей (студент, руководитель, сотрудник деканата)."
        ),
        "domain": "образовательные процессы в вузе",
    },
    {
        "id": "edo_department",
        "name": "Подсистема ЭДО кафедры",
        "description": (
            "Разработка подсистемы электронного документооборота (ЭДО) "
            "для кафедры в составе корпоративной информационной системы вуза. "
            "Подсистема должна обеспечивать регистрацию, согласование и "
            "хранение служебных записок, приказов, заявлений и других "
            "документов, а также поддерживать контроль сроков исполнения "
            "и разграничение прав доступа."
        ),
        "domain": "электронный документооборот",
    },
]

print(f"Количество тестовых проектов: {len(test_projects)}")
for project in test_projects:
    print(f"- {project['id']}: {project['name']}")

# ============================================================
# Заготовки для локальных текстов:
#  - фрагменты ГОСТ 19.201-78;
#  - фрагменты методических материалов / шаблонов МУИВ.
# ------------------------------------------------------------
# В учебном примере мы используем ПЛЕЙСХОЛДЕРЫ.
# На практике сюда можно:
#  - либо вставить выдержки из ГОСТ/методичек вручную,
#  - либо считать их из файлов (txt / docx / pdf).
# ============================================================

# ВНИМАНИЕ:
# Из-за ограничений по авторским правам сюда НЕЛЬЗЯ вставлять полный текст ГОСТ.
# Мы работаем только с фрагментами и собственными конспектами.

gost_text_raw = """
[ГОСТ 19.201-78, условный конспект]

Настоящий стандарт устанавливает содержание и порядок оформления
технического задания на разработку автоматизированных систем и
программных средств.

Техническое задание должно включать следующие разделы:
— Введение;
— Основания для разработки;
— Назначение системы;
— Требования к системе (функциональные, надежности, безопасности и т.д.);
— Состав и содержание работ по созданию системы;
— Порядок контроля и приемки;
— Требования к подготовке объекта автоматизации к вводу системы в действие;
— Требования к документированию;
— Источники разработки.

Каждый раздел должен содержать конкретные формулировки, необходимые
для разработки, внедрения и эксплуатации системы.
"""

muiv_methodology_text_raw = """
[Методические рекомендации МУИВ, условный текст]

В техническом задании на разработку информационной системы для нужд
университета необходимо учитывать особенности организационной структуры
ЧОУ ВО «Московский университет имени С. Ю. Витте».

В разделе «Основания для разработки» отражаются нормативные документы
университета, приказы, решения ученого совета, а также результаты анализа
существующих проблем в предметной области.

В разделе «Назначение системы» описываются цели создания системы,
основные функции, основные категории пользователей (администратор системы,
сотрудники кафедр, сотрудники деканатов, сотрудники ИТО и т.д.).

В разделе «Требования к системе» указываются:
— функциональные требования (перечень функций и операций);
— требования к надежности, безопасности и защите персональных данных;
— требования к интерфейсу и удобству работы пользователей;
— требования к интеграции с действующими информационными системами вуза.
"""

local_docs = [
    {
        "id": "gost_19_201_78",
        "source_type": "gost",
        "title": "Конспект ГОСТ 19.201-78",
        "text": gost_text_raw,
    },
    {
        "id": "muiv_methodology",
        "source_type": "muiv",
        "title": "Методические рекомендации МУИВ",
        "text": muiv_methodology_text_raw,
    },
]

print(f"Количество локальных документов: {len(local_docs)}")
for doc in local_docs:
    print(f"- {doc['id']}: {doc['title']} (тип: {doc['source_type']})")

# ============================================================
# Используем тестовые KEY и SEARCH_ENGINE_ID (cx), заданные ранее.
# Функция принимает текстовый запрос и возвращает список результатов
# (title + link). Дальше ссылки будем использовать для загрузки HTML.
# ============================================================

def google_custom_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Выполняет поиск через Google Custom Search API и возвращает
    список результатов в виде словарей:
    {
        "title": <заголовок>,
        "link": <URL>
    }
    """
    endpoint = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": min(max(num_results, 1), 10),  # ограничение API: максимум 10 за раз
        "hl": "ru",  # русский интерфейс/результаты, где возможно
    }

    print(f"[GoogleSearch] Запрос: {query!r}")

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[GoogleSearch] Ошибка при запросе: {e}")
        return []

    items = data.get("items", []) or []
    results: List[Dict[str, Any]] = []

    for item in items:
        title = item.get("title") or ""
        link = item.get("link") or ""
        if not link:
            continue
        results.append(
            {
                "title": title,
                "link": link,
            }
        )

    print(f"[GoogleSearch] Найдено результатов: {len(results)}")
    return results

# ============================================================
# Здесь мы формируем несколько поисковых запросов, связанных с ТЗ
# на автоматизированные системы, и собираем результаты (URL + заголовок).
# ============================================================

search_queries = [
    '"техническое задание" "автоматизированная система" "ГОСТ 19.201"',
    '"техническое задание" "информационная система" вуз',
    '"техническое задание" разработка программного обеспечения ГОСТ',
]

web_search_results: List[Dict[str, Any]] = []

for q in search_queries:
    results = google_custom_search(q, num_results=5)
    web_search_results.extend(
        {
            "query": q,
            "title": item["title"],
            "link": item["link"],
        }
        for item in results
    )

# Уберем дубли по ссылкам
seen_links = set()
unique_web_search_results = []
for item in web_search_results:
    if item["link"] in seen_links:
        continue
    seen_links.add(item["link"])
    unique_web_search_results.append(item)

web_search_results = unique_web_search_results

print("\nИтоговый список найденных ссылок:")
for idx, item in enumerate(web_search_results, start=1):
    print(f"{idx:2d}. {item['title']}\n    {item['link']}")

print("\nВсего уникальных ссылок:", len(web_search_results))

# ============================================================
# Используем BeautifulSoup:
#  - удаляем <script>, <style> и подобные элементы;
#  - получаем текст из <body> (или всего документа, если нужно);
#  - приводим пробелы к аккуратному виду.
# ============================================================

def fetch_and_extract_text(url: str, timeout: int = 15) -> str:
    """
    Загружает HTML-страницу по URL и извлекает из неё текстовое содержимое.
    Возвращает строку (возможно, довольно длинную).
    При ошибках возвращает пустую строку.
    """
    print(f"[Fetch] Загружаем: {url}")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()

        # Пытаемся угадать кодировку, если не указана
        if not resp.encoding:
            resp.encoding = resp.apparent_encoding

        html = resp.text
    except Exception as e:
        print(f"[Fetch] Ошибка при загрузке страницы: {e}")
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Удаляем скрипты, стили и подобное "мусорное" содержимое
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        # Пытаемся взять текст из <body>, если он есть
        body = soup.body
        if body is not None:
            text = body.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")

        # Нормализуем пробелы и пустые строки
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]  # убираем пустые строки
        cleaned_text = "\n".join(lines)

        return cleaned_text
    except Exception as e:
        print(f"[Fetch] Ошибка при парсинге HTML: {e}")
        return ""

# ============================================================
# Результат: список словарей web_docs:
#  {
#    "id": ...,
#    "source_type": "web_example",
#    "title": ...,
#    "url": ...,
#    "text": <извлечённый текст>
#  }
# Этот список позже будет участвовать в формировании корпуса RAG.
# ============================================================

web_docs: List[Dict[str, Any]] = []

for idx, item in enumerate(web_search_results, start=1):
    url = item["link"]
    title = item["title"]

    text = fetch_and_extract_text(url)
    if not text:
        print(f"[WebDoc] Пустой текст, пропускаем: {url}")
        continue

    doc = {
        "id": f"web_example_{idx}",
        "source_type": "web_example",
        "title": title,
        "url": url,
        "text": text,
    }
    web_docs.append(doc)

print("\nКоличество успешно загруженных веб-документов:", len(web_docs))
for doc in web_docs:
    print(f"- {doc['id']}: {doc['title']}")

# ============================================================
# Объединяем локальные документы и веб-документы в единый список.
# На следующем шаге (в блоке 4) будем:
#  - нормализовать текст;
#  - разбивать его на фрагменты (chunks);
#  - формировать RAG-файлы.
# ============================================================

all_source_docs: List[Dict[str, Any]] = []

# Добавляем локальные документы (ГОСТ + методички)
all_source_docs.extend(local_docs)

# Добавляем веб-документы (примерные ТЗ)
all_source_docs.extend(web_docs)

print("Итоговое количество документов в корпусе:", len(all_source_docs))

# Для удобства можно посмотреть короткую табличку
df_sources = pd.DataFrame(
    [
        {
            "id": doc["id"],
            "source_type": doc["source_type"],
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "text_preview": (doc["text"][:200] + "…") if len(doc["text"]) > 200 else doc["text"],
        }
        for doc in all_source_docs
    ]
)

df_sources

# ============================================================
# Мы НЕ делаем сложную NLP-очистку, только:
#  - убираем лишние пробелы и пустые строки;
#  - фильтруем заведомо мусорные случаи.
# ============================================================

import re

def looks_like_broken_or_binary(text: str) -> bool:
    """
    Простая эвристика: пытаемся понять, что текст выглядит как
    бинарный мусор (обрывок PDF, неправильная кодировка и т.п.).
    Возвращает True, если текст стоит отбросить полностью.
    """
    if not text:
        return True

    stripped = text.lstrip()

    # Явный признак PDF-файла
    if stripped.startswith("%PDF-"):
        return True

    # Частый паттерн "сломанной" кириллицы (Ð, Ñ и т.п.)
    # Если в первых 200 символах много таких символов — считаем мусором.
    prefix = stripped[:200]
    broken_chars = sum(ch in "ÐÑ�" for ch in prefix)
    if broken_chars > 10:  # порог подбираем грубо
        return True

    # Если текст почти целиком состоит из небуквенно-цифровых символов
    sample = stripped[:500]
    if sample:
        alpha_num = sum(ch.isalnum() for ch in sample)
        if alpha_num / len(sample) < 0.2:
            return True

    return False


def normalize_text(text: str) -> str:
    """
    Нормализует текст:
    - убирает лишние пробелы и пустые строки;
    - заменяет последовательности пробелов на один пробел внутри строк.
    """
    if not text:
        return ""

    # Унифицируем переводы строк
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Разбиваем на строки, чистим каждую
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Заменяем множественные пробелы на один
        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)

    # Собираем обратно с переводами строк
    cleaned = "\n".join(lines)

    return cleaned


# Проверочный пример (по желанию)
for doc in all_source_docs[:3]:
    print(f"\nДокумент {doc['id']} ({doc['source_type']}):")
    print("Исходный размер текста:", len(doc["text"]))
    print("Похоже на мусор?:", looks_like_broken_or_binary(doc["text"]))

# ============================================================
# Стратегия:
#  - сначала делим текст по абзацам (по переводу строк);
#  - затем собираем абзацы в чанки так, чтобы
#    длина по символам была в разумных пределах.
# ============================================================

from typing import Tuple

def split_text_into_chunks(
    text: str,
    max_chars: int = 800,
    min_chars: int = 300,
) -> List[str]:
    """
    Разбивает нормализованный текст на чанки длиной от min_chars до max_chars.
    Чанки формируются из абзацев (строк), чтобы не резать текст посередине фраз.
    """
    if not text:
        return []

    paragraphs = text.split("\n")
    chunks: List[str] = []
    current_chunk_lines: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # +1 за потенциальный перевод строки
        added_len = len(para) + (1 if current_chunk_lines else 0)

        if current_len + added_len <= max_chars:
            # Просто добавляем абзац в текущий чанк
            current_chunk_lines.append(para)
            current_len += added_len
        else:
            # Если текущий чанк достаточно длинный — сохраняем его
            if current_chunk_lines and current_len >= min_chars:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = [para]
                current_len = len(para)
            else:
                # Иначе пытаемся всё равно добить текущий чанк
                # (может он совсем пустой/короткий)
                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = [para]
                current_len = len(para)

    # Добавляем последний чанк, если он не пустой
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        # Если последний чанк совсем крошечный, можно попробовать
        # слить его с предыдущим — здесь для простоты просто сохраняем.
        chunks.append(chunk_text)

    # На всякий случай убираем совсем маленькие кусочки
    chunks = [c for c in chunks if len(c) >= 100]

    return chunks


# Небольшая проверка на локальных документах
for doc in local_docs:
    norm = normalize_text(doc["text"])
    small_chunks = split_text_into_chunks(norm)
    print(f"\nДокумент {doc['id']}:")
    print("  Нормализованный размер:", len(norm))
    print("  Число чанков:", len(small_chunks))
    if small_chunks:
        print("  Пример чанка:\n", small_chunks[0][:200], "...")

# ============================================================
# Для каждого исходного документа:
#   1) проверяем, не является ли он "битым";
#   2) нормализуем текст;
#   3) разбиваем на чанки;
#   4) для каждого чанка создаём запись с метаданными.
# ============================================================

corpus_chunks: List[Dict[str, Any]] = []
chunk_id_counter = 0

for doc in all_source_docs:
    doc_id = doc["id"]
    source_type = doc["source_type"]
    title = doc.get("title", "")
    url = doc.get("url", "")

    raw_text = doc["text"]

    # Фильтр "битых" документов
    if looks_like_broken_or_binary(raw_text):
        print(f"[CHUNKS] Документ {doc_id} ({source_type}) отфильтрован как мусор.")
        continue

    # Нормализация
    normalized = normalize_text(raw_text)

    # Если после нормализации текст слишком короткий — пропускаем
    if len(normalized) < 200:
        print(f"[CHUNKS] Документ {doc_id} ({source_type}) слишком короткий после очистки, пропускаем.")
        continue

    # Разбиение на чанки
    chunks = split_text_into_chunks(normalized, max_chars=800, min_chars=300)

    if not chunks:
        print(f"[CHUNKS] Для документа {doc_id} ({source_type}) не удалось сформировать чанки.")
        continue

    for idx, chunk_text in enumerate(chunks):
        chunk_id_counter += 1
        corpus_chunks.append(
            {
                "chunk_id": f"chunk_{chunk_id_counter}",
                "doc_id": doc_id,
                "source_type": source_type,
                "title": title,
                "url": url,
                "chunk_index": idx,
                "text": chunk_text,
            }
        )

print("\nИтоговое количество чанков в корпусе:", len(corpus_chunks))

# Сводный DataFrame по чанкам
df_chunks = pd.DataFrame(
    [
        {
            "chunk_id": ch["chunk_id"],
            "doc_id": ch["doc_id"],
            "source_type": ch["source_type"],
            "title": ch["title"],
            "chunk_index": ch["chunk_index"],
            "text_preview": (ch["text"][:200] + "…") if len(ch["text"]) > 200 else ch["text"],
        }
        for ch in corpus_chunks
    ]
)

df_chunks.head()

# ============================================================
# Форматы:
#  - JSONL (по строке на чанк) — удобно читать по одному объекту;
#  - CSV — удобно смотреть в табличном виде.
# Эти файлы потом можно использовать во Flask-приложении.
# ============================================================

output_dir = Path("rag_corpus")
output_dir.mkdir(exist_ok=True)

jsonl_path = output_dir / "rag_corpus_chunks.jsonl"
csv_path = output_dir / "rag_corpus_chunks.csv"

# Сохранение JSONL
with jsonl_path.open("w", encoding="utf-8") as f:
    for ch in corpus_chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")

# Сохранение CSV
df_chunks.to_csv(csv_path, index=False, encoding="utf-8")

print("Корпус чанков сохранён:")
print(" - JSONL:", jsonl_path)
print(" - CSV:   ", csv_path)

# ============================================================
# Используем sklearn.TfidfVectorizer:
#  - каждый чанк (fragment) превращаем в TF–IDF-вектор;
#  - затем сможем быстро вычислять косинусное сходство с запросами.
# ============================================================

# Собираем тексты чанков в список в том же порядке, в котором они
# хранятся в corpus_chunks. Индекс в этом списке будет совпадать
# с индексом строки в TF–IDF-матрице.
chunk_texts: List[str] = [ch["text"] for ch in corpus_chunks]

print("Количество чанков для TF–IDF:", len(chunk_texts))

# Настройка TF–IDF-векторизатора
# Параметры можно тюнить:
#  - max_features ограничивает словарь (для контроля памяти);
#  - ngram_range=(1, 2) учитывает униграммы и биграммы.
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
)

# Обучение векторизатора на корпусе и получение матрицы признаков
tfidf_matrix = tfidf_vectorizer.fit_transform(chunk_texts)

print("Форма TF–IDF-матрицы:", tfidf_matrix.shape)

# ============================================================
# Параметры:
#  - query_text: текстовый запрос (описание проекта / раздела ТЗ);
#  - top_k: сколько фрагментов вернуть;
#  - allowed_source_types: фильтр по типу источников
#    (например, ["gost", "muiv"] или None для всех).
#
# Возвращает:
#  - список словарей с полями:
#      chunk_id, doc_id, source_type, title, score, text
# ============================================================

def retrieve_chunks(
    query_text: str,
    top_k: int = 5,
    allowed_source_types: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Ищет наиболее релевантные чанки для заданного текстового запроса.
    Можно ограничить поиск по типу источников (gost, muiv, web_example).

    Возвращает список чанков с метаданными и оценкой сходства (score).
    """
    if not query_text.strip():
        raise ValueError("Пустой запрос: query_text не должен быть пустым.")

    # Векторизуем запрос (одна строка)
    query_vec = tfidf_vectorizer.transform([query_text])

    # Вычисляем косинусное сходство запрос↔все чанки
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]  # shape: (n_chunks,)

    # Если задан фильтр по типу источников — отбрасываем лишние
    if allowed_source_types is not None:
        allowed_source_types_set = set(allowed_source_types)
        # Заменяем сходство на -1 для чанков с неподходящим source_type
        for i, ch in enumerate(corpus_chunks):
            if ch["source_type"] not in allowed_source_types_set:
                similarities[i] = -1.0

    # Находим индексы top_k чанков по убыванию сходства
    # argsort даёт индексы по возрастанию, поэтому переворачиваем
    top_k = max(1, min(top_k, len(corpus_chunks)))
    top_indices = similarities.argsort()[::-1][:top_k]

    results: List[Dict[str, Any]] = []

    for idx in top_indices:
        score = float(similarities[idx])
        ch = corpus_chunks[idx]
        results.append(
            {
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],
                "source_type": ch["source_type"],
                "title": ch["title"],
                "url": ch.get("url", ""),
                "chunk_index": ch["chunk_index"],
                "score": score,
                "text": ch["text"],
            }
        )

    return results

# ============================================================
# Проверяем три сценария:
#  1) Поиск только по ГОСТ (RAG-GOST);
#  2) Поиск по ГОСТ + методическим материалам (локальный контекст);
#  3) Поиск по всему корпусу (включая web_example).
# ============================================================

test_query = (
    "Техническое задание на разработку системы учета сотрудников "
    "в университете, требуется описать назначение системы и основные функции."
)

print("Тестовый запрос:")
print(test_query)
print("\n--- Только ГОСТ (source_type=['gost']) ---")

results_gost = retrieve_chunks(
    query_text=test_query,
    top_k=5,
    allowed_source_types=["gost"],
)

for r in results_gost:
    print(f"\n[{r['source_type']}] {r['title']} (score={r['score']:.3f})")
    print(r["text"][:300], "...")


print("\n\n--- ГОСТ + методички (source_type=['gost', 'muiv']) ---")

results_local = retrieve_chunks(
    query_text=test_query,
    top_k=5,
    allowed_source_types=["gost", "muiv"],
)

for r in results_local:
    print(f"\n[{r['source_type']}] {r['title']} (score={r['score']:.3f})")
    print(r["text"][:300], "...")


print("\n\n--- Весь корпус (без фильтра по типу) ---")

results_full = retrieve_chunks(
    query_text=test_query,
    top_k=5,
    allowed_source_types=None,  # None = все источники
)

for r in results_full:
    print(f"\n[{r['source_type']}] {r['title']} (score={r['score']:.3f})")
    print(r["text"][:300], "...")

# ============================================================
# Функция llm_chat_completion:
#  - принимает системный промпт, текст пользователя и имя модели;
#  - возвращает текст ответа ассимистента;
#  - при ошибках печатает сообщение и возвращает заглушку.
# ============================================================

from typing import Optional

def llm_chat_completion(
    system_prompt: str,
    user_prompt: str,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> str:
    """
    Вызывает LLM через Proxy API ChatGPT и возвращает текст ответа.

    Параметры:
    - system_prompt: роль и стиль модели (инструкции эксперту по ГОСТ);
    - user_prompt: задание пользователя (описание проекта + раздел ТЗ);
    - model_name: имя модели (если None, используется DEFAULT_LLM_MODEL);
    - temperature: "креативность" ответа;
    - max_tokens: ограничение на длину ответа (если поддерживается API).
    """
    if model_name is None:
        model_name = DEFAULT_LLM_MODEL

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            # max_tokens здесь может игнорироваться конкретной моделью,
            # но мы оставляем параметр как подсказку.
        )
        # Для OpenAI-совместимого API предполагаем такой интерфейс:
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"[LLM ERROR] Ошибка при вызове модели: {e}")
        return (
            "Не удалось получить ответ от модели. "
            "Проверьте настройки Proxy API и повторите попытку."
        )

# ============================================================
# Разносим логику по отдельным функциям, чтобы потом легко менять
# стиль/формат без правок во всех местах.
# ============================================================

def build_system_prompt_for_tz_section() -> str:
    """
    Формирует системный промпт для роли LLM:
    модель выступает экспертом по ГОСТ 19.201-78 и
    методическим материалам университета.
    """
    system_prompt = (
        "Ты — эксперт по стандартизации и проектированию информационных систем.\n"
        "Твоя задача — формировать разделы технического задания (ТЗ) в соответствии "
        "с ГОСТ 19.201-78 и методическими материалами университета.\n\n"
        "Требования к ответу:\n"
        "1. Пиши по-русски, академичным, но понятным языком.\n"
        "2. Соблюдай структуру и терминологию ГОСТ 19.201-78.\n"
        "3. Учитывай, что объект автоматизации — информационная система в вузе.\n"
        "4. Не выдумывай факты о конкретном университете, если они не указаны "
        "во входных данных или контексте.\n"
        "5. Формируй текст так, чтобы его можно было сразу вставить в ТЗ.\n"
    )
    return system_prompt


def build_user_prompt_for_tz_section(
    project: Dict[str, Any],
    section_name: str,
    extra_instructions: Optional[str] = None,
) -> str:
    """
    Формирует пользовательский промпт (часть user) для генерации
    конкретного раздела ТЗ по заданному проекту.

    project: словарь из test_projects (id, name, description, domain)
    section_name: название раздела ТЗ (например, 'Назначение системы').
    """
    project_name = project.get("name", "Информационная система")
    project_desc = project.get("description", "")
    project_domain = project.get("domain", "")

    base_prompt = (
        f"Проект: {project_name}\n"
        f"Предметная область: {project_domain}\n\n"
        f"Краткое описание проекта:\n{project_desc}\n\n"
        f"Необходимо сформировать раздел технического задания (ТЗ): «{section_name}».\n\n"
        "Опиши данный раздел так, как это принято в ГОСТ 19.201-78, с учётом того, "
        "что система создаётся для вуза. Следи за связностью текста и логикой "
        "изложения, избегай излишней воды и общих фраз."
    )

    if extra_instructions:
        base_prompt += "\n\nДополнительные требования:\n" + extra_instructions

    return base_prompt

# ============================================================
# Возвращает словарь с:
#  - project_id, section_name, mode, model_name, text, used_chunks.
# ============================================================

def generate_tz_section(
    project: Dict[str, Any],
    section_name: str,
    mode: str = "baseline",
    model_name: Optional[str] = None,
    top_k_chunks: int = 8,
) -> Dict[str, Any]:
    """
    Генерирует текст раздела ТЗ для заданного проекта и режима работы.

    Параметры:
    - project: словарь из test_projects;
    - section_name: строка, название раздела ТЗ;
    - mode:
        * 'baseline'  — без использования RAG;
        * 'rag_gost'  — использовать только чанки из ГОСТ (source_type='gost');
        * 'rag_full'  — использовать весь корпус (gost + muiv + web_example);
    - model_name: имя модели Proxy API (если None — DEFAULT_LLM_MODEL);
    - top_k_chunks: сколько фрагментов подставлять в контекст.

    Результат:
    - словарь с ключами:
        project_id, section_name, mode, model_name, text, used_chunks (список chunk_id).
    """
    project_id = project.get("id", "unknown_project")

    # 1. Определяем, какие источники использовать для RAG
    allowed_source_types = None
    if mode == "rag_gost":
        allowed_source_types = ["gost"]
    elif mode == "rag_full":
        allowed_source_types = None  # все источники
    elif mode == "baseline":
        allowed_source_types = None
    else:
        raise ValueError(f"Неизвестный режим mode={mode!r}")

    # 2. Подбор релевантного контекста (если не baseline)
    retrieved_chunks: List[Dict[str, Any]] = []

    if mode in ("rag_gost", "rag_full"):
        # Формируем запрос для поиска: описание проекта + название раздела
        query_text = (
            f"{project.get('description', '')}\n"
            f"Раздел ТЗ: {section_name}.\n"
            "Техническое задание на разработку информационной системы."
        )

        retrieved_chunks = retrieve_chunks(
            query_text=query_text,
            top_k=top_k_chunks,
            allowed_source_types=allowed_source_types,
        )

    # 3. Строим системный и пользовательский промпты
    system_prompt = build_system_prompt_for_tz_section()

    base_user_prompt = build_user_prompt_for_tz_section(
        project=project,
        section_name=section_name,
        extra_instructions=None,
    )

    # 4. Добавляем контекст (если есть retriever-результаты)
    if retrieved_chunks:
        context_parts = []
        for i, ch in enumerate(retrieved_chunks, start=1):
            # Для экономии токенов можем слегка обрезать слишком длинные фрагменты
            chunk_text = ch["text"]
            if len(chunk_text) > 800:
                chunk_text = chunk_text[:800] + "…"

            context_parts.append(
                f"[Фрагмент {i} | источник: {ch['source_type']} | документ: {ch['title']}]\n"
                f"{chunk_text}"
            )

        context_block = "\n\n".join(context_parts)

        user_prompt = (
            base_user_prompt
            + "\n\n--- Контекст (фрагменты из ГОСТ и связанных документов) ---\n"
            + context_block
            + "\n\nИспользуя приведённый контекст, сформируй связный и аккуратный текст "
              "раздела ТЗ. При необходимости переформулируй фрагменты, не копируй их дословно."
        )
    else:
        # Baseline или случай, когда retriever ничего не нашёл
        user_prompt = (
            base_user_prompt
            + "\n\nКонтекст по ГОСТ и методическим материалам не подставляется. "
              "Ориентируйся на общие требования к ТЗ в сфере разработки "
              "информационных систем, но соблюдай структуру ГОСТ 19.201-78."
        )

    # 5. Вызываем модель
    answer_text = llm_chat_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        temperature=0.2,
    )

    result = {
        "project_id": project_id,
        "section_name": section_name,
        "mode": mode,
        "model_name": model_name or DEFAULT_LLM_MODEL,
        "text": answer_text,
        "used_chunks": [ch["chunk_id"] for ch in retrieved_chunks],
    }

    return result
