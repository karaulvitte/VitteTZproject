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
