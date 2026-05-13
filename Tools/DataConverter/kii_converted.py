import openpyxl

# Загружаем Excel с данными КИИ
dataframe = openpyxl.load_workbook("kii_data.xlsx")
data = dataframe.active

convertedWb = openpyxl.Workbook()
convSt = convertedWb.active

# Заголовки — все признаки объекта КИИ
convSt.append([
    "object_name",           # A - наименование
    "sector",                # B - сектор
    "level",                 # C - уровень
    "service_scale",         # D - масштаб услуг
    "process_criticality",   # E - критичность
    "num_users",             # F - пользователи
    "num_territories",       # G - территории
    "predicted_financial_damage", # H - финущерб
    "recovery_time_hours",   # I - время восстановления
    "critical_processes_count", # J - крит. процессы
    "integrations_count",    # K - интеграции
    "personal_data_subjects", # L - субъекты ПДн
    "affected_employees",    # M - затронутые сотрудники
    "continuous_operation",  # N - непрерывный режим
    "uses_automated_control_system", # O - АСУ ТП
    "provides_gov_services", # P - госуслуги
    "life_health_impact",    # Q - жизнь и здоровье
    "ecological_impact",     # R - экология
    "defense_impact",        # S - оборона
    "public_order_impact",   # T - общ. порядок
    "transport_impact",      # U - транспорт
    "communication_impact",  # V - связь
    "sensitive_info",        # W - чувств. информация
    "category_level"         # X - целевая метка (0-3)
])

# === КОНВЕРТАЦИЯ ===

# Сектор (B) — текстовые значения, кодируем числами
SECTOR_MAP = {
    "Энергетика": 0, "Транспорт": 1, "Связь": 2, "Здравоохранение": 3,
    "Банковская сфера": 4, "Оборонная промышленность": 5,
    "Государственное управление": 6, "Наука": 7, "Топливная промышленность": 8
}

# Уровень (C)
LEVEL_MAP = {"Федеральный": 0, "Региональный": 1, "Муниципальный": 2, "Объектовый": 3}

# Масштаб (D)
SCALE_MAP = {"Вся страна": 0, "Федеральный округ": 1, "Субъект РФ": 2, 
             "Несколько муниципалитетов": 3, "Один город": 4}

# Критичность (E)
CRITICALITY_MAP = {"Критическая": 0, "Высокая": 1, "Средняя": 2, "Низкая": 3}

# Обрабатываем строки 4-123 (120 объектов)
for row_idx in range(4, 124):
    out_row = row_idx - 2  # Выходная строка (начиная с 2, т.к. 1 — заголовки)
    
    # A — наименование (оставляем как есть)
    convSt[f'A{out_row}'] = data[f'A{row_idx}'].value
    
    # B — сектор (кодируем)
    sector_val = data[f'B{row_idx}'].value
    convSt[f'B{out_row}'] = SECTOR_MAP.get(sector_val, -1)
    
    # C — уровень
    level_val = data[f'C{row_idx}'].value
    convSt[f'C{out_row}'] = LEVEL_MAP.get(level_val, -1)
    
    # D — масштаб услуг
    scale_val = data[f'D{row_idx}'].value
    convSt[f'D{out_row}'] = SCALE_MAP.get(scale_val, -1)
    
    # E — критичность
    crit_val = data[f'E{row_idx}'].value
    convSt[f'E{out_row}'] = CRITICALITY_MAP.get(crit_val, -1)
    
    # F-M — числовые параметры (оставляем как есть)
    for col_letter in ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']:
        convSt[f'{col_letter}{out_row}'] = data[f'{col_letter}{row_idx}'].value
    
    # N-W — бинарные признаки (+ → 1, пусто → 0)
    for col_letter in ['N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']:
        val = data[f'{col_letter}{row_idx}'].value
        convSt[f'{col_letter}{out_row}'] = 1 if val == '+' else 0
    
    # X — целевая метка (категория)
    convSt[f'X{out_row}'] = data[f'X{row_idx}'].value

convertedWb.save("kii_converted.xlsx")
print("Конвертация завершена! Создан файл kii_converted.xlsx")
