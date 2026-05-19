# KII Significance Service

Проект определяет категорию значимости объекта КИИ и состоит из двух сервисов:

- `inference-api` на `FastAPI` рассчитывает итоговую категорию и отдает REST API.
- `dashboard` на `Streamlit` предоставляет веб-интерфейс для ввода данных и просмотра результатов.

Дополнительно есть каталог `model-training` со скриптами обучения и проверки модели.

## Структура проекта

```text
KIIIncidentAIService/
  README.md
  docker-compose.yml
  inference-api/
    app.py
    Dockerfile
    requirements.txt
    models/
      model.keras
      model.h5
  dashboard/
    app.py
    Dockerfile
    requirements.txt
    data/
      objects.json
      incidents.json
  model-training/
    finetune_kii.py
    validate_inference.py
    retrain_model.py
    generate_synthetic_data.py
    requirements.txt
    sitecustomize.py
    synthetic_kii_data.csv
  Tools/
    DataConverter/
      converter.py
      synthetic_kii_data.py
      kii_converted.py
      convert.xlsx
      Converted.xlsx
```

## Для чего нужен каждый файл

### Корень проекта

- `README.md` - инструкция по запуску, обучению и структуре проекта.
- `docker-compose.yml` - поднимает `inference-api` и `dashboard` одной командой.

### `inference-api/`

- `inference-api/app.py` - основной FastAPI-сервис: загрузка модели, `/health`, `/predict`, выбор модели и гибридная логика `heuristics + ML`.
- `inference-api/Dockerfile` - контейнер API.
- `inference-api/requirements.txt` - Python-зависимости API.
- `inference-api/models/model.keras` - основной файл обученной модели для inference.
- `inference-api/models/model.h5` - резервная копия модели в формате H5.

### `dashboard/`

- `dashboard/app.py` - Streamlit-приложение для работы с API и отображения данных.
- `dashboard/Dockerfile` - контейнер dashboard.
- `dashboard/requirements.txt` - зависимости dashboard.
- `dashboard/data/objects.json` - локальный JSON-реестр объектов КИИ для интерфейса.
- `dashboard/data/incidents.json` - локальные данные по инцидентам для интерфейса.

### `model-training/`

- `model-training/finetune_kii.py` - основной скрипт дообучения модели, совместимой с `inference-api`.
- `model-training/validate_inference.py` - локальная проверка, что API реально загружает сохраненную модель и использует ее в `/predict`.
- `model-training/retrain_model.py` - альтернативный скрипт полного переобучения модели и сохранения дополнительных артефактов.
- `model-training/generate_synthetic_data.py` - генерация синтетического CSV-датасета для обучения.
- `model-training/requirements.txt` - зависимости для обучения и валидации модели.
- `model-training/sitecustomize.py` - локальный патч совместимости сохранения `.keras` для используемой версии `tf.keras`.
- `model-training/synthetic_kii_data.csv` - синтетический датасет для обучения и валидации.

### `Tools/DataConverter/`

- `Tools/DataConverter/converter.py` - утилита конвертации Excel-данных в пригодную для обработки структуру.
- `Tools/DataConverter/synthetic_kii_data.py` - вспомогательный скрипт подготовки или преобразования данных для обучения.
- `Tools/DataConverter/kii_converted.py` - вспомогательный скрипт работы с уже преобразованными данными.
- `Tools/DataConverter/convert.xlsx` - исходный Excel-файл для конвертации.
- `Tools/DataConverter/Converted.xlsx` - результат конвертации Excel-данных.

## Как устроена модель

- API использует гибридный подход: эвристический скоринг плюс нейросетевая классификация.
- Каноническое место хранения модели для сервиса: `inference-api/models/`.
- Основной формат: `model.keras`.
- Резервный формат: `model.h5`.

Категории:

- `0` - `Без категории`
- `1` - `Третья категория`
- `2` - `Вторая категория`
- `3` - `Первая категория`

## Подготовка локального окружения

Команды ниже рассчитаны на `PowerShell` и выполняются из корня проекта.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\model-training\requirements.txt
pip install -r .\inference-api\requirements.txt
pip install httpx
```

`httpx` нужен для `fastapi.testclient`, который использует `model-training/validate_inference.py`.

## Обучение модели

Если датасет уже есть, можно сразу запускать дообучение:

```powershell
cd .\model-training
python .\finetune_kii.py
cd ..
```

Если нужен новый синтетический датасет:

```powershell
cd .\model-training
python .\generate_synthetic_data.py
python .\finetune_kii.py
cd ..
```

После успешного обучения ожидаются файлы:

- `inference-api/models/model.keras`
- `inference-api/models/model.h5`

## Проверка модели перед запуском сервиса

После обучения выполните:

```powershell
python .\model-training\validate_inference.py
```

Успешная проверка выглядит так:

- `model_loaded: true`
- `model_usable: true`
- `model_role: supporting_classifier`
- в конце выводится `ПРОВЕРКА ПРОЙДЕНА: API загрузил и использует обученную модель.`

## Запуск сервиса через Docker Compose

Если модель уже лежит в `inference-api/models/`, сервис можно поднимать сразу:

```powershell
docker compose up --build
```

Запуск в фоне:

```powershell
docker compose up --build -d
```

Остановка:

```powershell
docker compose down
```

После старта сервисы доступны по адресам:

- Dashboard: `http://localhost:8501`
- Swagger UI API: `http://localhost:8008/docs`
- Health endpoint: `http://localhost:8008/health`

## Полный рекомендуемый сценарий запуска

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\model-training\requirements.txt
pip install -r .\inference-api\requirements.txt
pip install httpx
cd .\model-training
python .\finetune_kii.py
cd ..
python .\model-training\validate_inference.py
docker compose up --build -d
```

## Что делает `docker-compose.yml`

- собирает контейнер `inference-api` из `./inference-api`
- собирает контейнер `dashboard` из `./dashboard`
- публикует API на `8008`
- публикует dashboard на `8501`
- передает в API:
  - `MODEL_PATH=/app/models/model.keras`
  - `DEFAULT_MODEL_CHOICE=keras`
  - `ALLOW_MODEL_FALLBACK=false`

Если нужно принудительно использовать H5-модель, замените в `docker-compose.yml`:

```yaml
MODEL_PATH: /app/models/model.h5
DEFAULT_MODEL_CHOICE: h5
```

## Почему могли появиться дубли `model.h5` и `model.keras`

Штатное место хранения моделей только одно:

- `inference-api/models/model.keras`
- `inference-api/models/model.h5`

Если в проекте появились дополнительные копии в каталогах вроде:

- `models/`
- `model-training/models/`
- `inference-api/inference-api/models/`

это не часть нормального развёртывания. Такие дубли были временным побочным эффектом диагностического скрипта валидации, который раскладывал модель по нескольким относительным путям для поиска ошибки загрузки. Текущая версия `model-training/validate_inference.py` больше этого не делает, а `inference-api/app.py` теперь ищет модель относительно собственного файла.

Лишние копии можно удалить. Оставить нужно только файлы в `inference-api/models/`.

## Проверка состояния API после запуска

Быстрая проверка:

```powershell
curl http://localhost:8008/health
```

Если сервис использует модель, в ответе будут:

- `model_loaded: true`
- `model_usable: true`

Если модель не найдена, API все равно поднимется, но останется в эвристическом режиме.

## Ограничения demo-версии

- используется синтетический датасет
- вместо БД используется локальный JSON
- часть логики категорирования остается эвристической
- авторизация и ролевая модель не реализованы
