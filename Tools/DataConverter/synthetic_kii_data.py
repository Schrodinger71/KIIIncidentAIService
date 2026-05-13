import pandas as pd
import openpyxl

# Читаем CSV
df = pd.read_csv("../../model-training/synthetic_kii_data.csv")

# Создаём Excel в формате с плюсиками
wb = openpyxl.Workbook()
ws = wb.active

# Заголовки (как в исходных таблицах)
headers = [
    "Наименование", "Сектор", "Уровень", "Масштаб услуг", "Критичность",
    "Пользователи", "Территории", "Финущерб", "Восстановление",
    "Крит.процессы", "Интеграции", "Субъекты ПДн", "Сотрудники",
    "Непрерывный", "АСУ ТП", "Госуслуги", "Жизнь/здоровье",
    "Экология", "Оборона", "Общ.порядок", "Транспорт", "Связь",
    "Чувств.инфо", "Категория"
]
ws.append(headers)
ws.append([])  # пустая строка
ws.append([])  # пустая строка

# Данные
for _, row in df.iterrows():
    excel_row = [
        row['object_name'],
        row['sector'],
        row['level'],
        row['service_scale'],
        row['process_criticality'],
        row['num_users'],
        row['num_territories'],
        row['predicted_financial_damage'],
        row['recovery_time_hours'],
        row['critical_processes_count'],
        row['integrations_count'],
        row['personal_data_subjects'],
        row['affected_employees'],
        '+' if row['continuous_operation'] else '',
        '+' if row['uses_automated_control_system'] else '',
        '+' if row['provides_gov_services'] else '',
        '+' if row['life_health_impact'] else '',
        '+' if row['ecological_impact'] else '',
        '+' if row['defense_impact'] else '',
        '+' if row['public_order_impact'] else '',
        '+' if row['transport_impact'] else '',
        '+' if row['communication_impact'] else '',
        '+' if row['sensitive_info'] else '',
        row['category_level']
    ]
    ws.append(excel_row)

wb.save("kii_data.xlsx")
print("Создан kii_data.xlsx")
