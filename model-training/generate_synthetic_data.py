import numpy as np
import pandas as pd

# --- Настройки генерации ---
NUM_SAMPLES = 5000               # Сколько объектов создать
RANDOM_SEED = 42                 # Для воспроизводимости
np.random.seed(RANDOM_SEED)

# --- Списки значений для категориальных признаков ---
SECTORS = ['Энергетика', 'Транспорт', 'Связь', 'Здравоохранение', 'Банковская сфера', 
           'Оборонная промышленность', 'Государственное управление', 'Наука', 'Топливная промышленность']
LEVELS = ['Федеральный', 'Региональный', 'Муниципальный', 'Объектовый']
SERVICE_SCALES = ['Вся страна', 'Федеральный округ', 'Субъект РФ', 'Несколько муниципалитетов', 'Один город']
PROCESS_CRITICALITIES = ['Критическая', 'Высокая', 'Средняя', 'Низкая']

# --- Вспомогательные функции ---
def generate_object_name(i):
    prefixes = ['АСУ ТП', 'ИС', 'Платформа', 'Система', 'Сеть', 'Портал', 'Сервис']
    types = ['управления', 'мониторинга', 'обработки', 'передачи', 'хранения', 'аналитики']
    return f"{np.random.choice(prefixes)} {np.random.choice(types)} #{i+1}"

# Генерация числовых параметров с правдоподобными зависимостями
def generate_numeric_features(level_idx, sector_idx, criticality_idx, service_scale_idx):
    # level_idx: 0-Федеральный, 1-Региональный, 2-Муниципальный, 3-Объектовый
    # Чем выше уровень (меньше индекс), тем больше значения
    base_multiplier = max(4 - level_idx, 1)  # 4,3,2,1
    
    # Количество пользователей
    if level_idx == 0:
        num_users = int(np.random.triangular(50000, 200000, 2000000))
    elif level_idx == 1:
        num_users = int(np.random.triangular(5000, 50000, 500000))
    elif level_idx == 2:
        num_users = int(np.random.triangular(100, 5000, 100000))
    else:
        num_users = int(np.random.triangular(10, 500, 5000))

    # Количество территорий
    if service_scale_idx == 0:  # Вся страна
        num_territories = np.random.randint(50, 89)
    elif service_scale_idx == 1:  # Федеральный округ
        num_territories = np.random.randint(10, 30)
    elif service_scale_idx == 2:  # Субъект РФ
        num_territories = np.random.randint(1, 10)
    else:
        num_territories = np.random.randint(1, 5)

    # Финансовый ущерб (в рублях)
    if criticality_idx <= 1:  # Критическая/Высокая
        predicted_financial_damage = int(np.random.lognormal(mean=18, sigma=1.5))  # ~ сотни млн
    elif criticality_idx == 2:
        predicted_financial_damage = int(np.random.lognormal(mean=15, sigma=1.2))
    else:
        predicted_financial_damage = int(np.random.lognormal(mean=12, sigma=1.0))

    # Время восстановления (часы)
    if criticality_idx == 0:
        recovery_time = np.random.choice([0.5, 1, 2, 4])
    elif criticality_idx == 1:
        recovery_time = np.random.choice([4, 8, 12, 24])
    else:
        recovery_time = np.random.choice([24, 48, 72, 168])

    # Количество критичных процессов
    critical_processes_count = max(1, int(np.random.normal(loc=base_multiplier*3, scale=base_multiplier)))
    
    # Количество интеграций
    integrations_count = max(1, int(np.random.normal(loc=base_multiplier*5, scale=base_multiplier*2)))
    
    # Субъекты персональных данных
    personal_data_subjects = int(num_users * np.random.uniform(0.1, 1.5))
    
    # Затронутые сотрудники
    affected_employees = int(np.random.triangular(10, 100, 5000) * (base_multiplier/3))
    
    return (num_users, num_territories, predicted_financial_damage, recovery_time,
            critical_processes_count, integrations_count, personal_data_subjects, affected_employees)

# Генерация логических флагов с зависимостями
def generate_boolean_flags(sector, level_idx, criticality):
    # Оборонка и связь чаще имеют defense_impact и sensitive_info
    defense_impact = False
    if sector in ['Оборонная промышленность', 'Государственное управление']:
        defense_impact = np.random.choice([True, True, True, False])  # 75% True
    elif level_idx == 0 and criticality == 'Критическая':
        defense_impact = np.random.choice([True, False])
    
    # Непрерывный режим
    continuous_operation = criticality in ['Критическая', 'Высокая'] or np.random.rand() < 0.3
    
    # АСУ ТП
    uses_automated_control_system = sector in ['Энергетика', 'Транспорт', 'Топливная промышленность'] or np.random.rand() < 0.2
    
    # Госуслуги
    provides_gov_services = sector == 'Государственное управление' or np.random.rand() < 0.1
    
    # Жизнь и здоровье
    life_health_impact = sector in ['Здравоохранение', 'Транспорт'] or (criticality == 'Критическая' and np.random.rand() < 0.7)
    
    # Экология
    ecological_impact = sector in ['Энергетика', 'Топливная промышленность'] or np.random.rand() < 0.1
    
    # Общественный порядок
    public_order_impact = sector in ['Государственное управление', 'Транспорт'] or np.random.rand() < 0.15
    
    # Транспорт
    transport_impact = sector == 'Транспорт' or np.random.rand() < 0.1
    
    # Связь
    communication_impact = sector == 'Связь' or np.random.rand() < 0.1
    
    # Чувствительная информация
    sensitive_info = (defense_impact or provides_gov_services or sector in ['Оборонная промышленность', 'Банковская сфера']) and np.random.rand() < 0.9
    
    return (continuous_operation, uses_automated_control_system, provides_gov_services,
            life_health_impact, ecological_impact, defense_impact, public_order_impact,
            transport_impact, communication_impact, sensitive_info)

# --- Эвристический расчет категории (аналогично вашему API) ---
def calculate_category(row):
    # Упрощенная балльная система, отражающая логику Постановления №4
    score = 0.0
    # Социальный ущерб
    if row['life_health_impact']: score += 25
    if row['num_users'] > 1000000: score += 20
    elif row['num_users'] > 100000: score += 10
    elif row['num_users'] > 10000: score += 5
    if row['personal_data_subjects'] > 500000: score += 10
    elif row['personal_data_subjects'] > 100000: score += 5
    if row['num_territories'] > 20: score += 10
    elif row['num_territories'] > 5: score += 5
    
    # Экономический ущерб
    if row['predicted_financial_damage'] > 500_000_000: score += 25
    elif row['predicted_financial_damage'] > 100_000_000: score += 15
    elif row['predicted_financial_damage'] > 10_000_000: score += 5
    if row['recovery_time_hours'] <= 4: score += 15
    elif row['recovery_time_hours'] <= 24: score += 5
    if row['continuous_operation']: score += 5
    
    # Ущерб для обороны и безопасности
    if row['defense_impact']: score += 20
    if row['sensitive_info']: score += 10
    if row['public_order_impact']: score += 5
    if row['provides_gov_services']: score += 10
    
    # Отраслевые и технологические факторы
    if row['uses_automated_control_system']: score += 10
    if row['level'] == 'Федеральный': score += 10
    elif row['level'] == 'Региональный': score += 5
    if row['critical_processes_count'] > 5: score += 5
    if row['integrations_count'] > 10: score += 5
    
    # Определение категории по баллам (шкала соответствует значимости)
    if score >= 70:
        return 3  # Первая
    elif score >= 45:
        return 2  # Вторая
    elif score >= 20:
        return 1  # Третья
    else:
        return 0  # Без категории

# --- Основной цикл генерации ---
data = []
for i in range(NUM_SAMPLES):
    object_name = generate_object_name(i)
    sector = np.random.choice(SECTORS)
    level = np.random.choice(LEVELS, p=[0.15, 0.35, 0.25, 0.25])  # Распределение уровней
    service_scale = np.random.choice(SERVICE_SCALES)
    process_criticality = np.random.choice(PROCESS_CRITICALITIES, p=[0.1, 0.3, 0.4, 0.2])
    
    # Индексы для вспомогательных функций
    sector_idx = SECTORS.index(sector)
    level_idx = LEVELS.index(level)
    criticality_idx = PROCESS_CRITICALITIES.index(process_criticality)
    scale_idx = SERVICE_SCALES.index(service_scale)
    
    # Числовые признаки
    num_users, num_territories, fin_damage, rec_time, crit_proc, integr, pers_data, aff_emp = \
        generate_numeric_features(level_idx, sector_idx, criticality_idx, scale_idx)
    
    # Булевы признаки
    (continuous_op, uses_asu, gov_services, life_health, eco, defense, pub_order,
     transport, comm, sensitive) = generate_boolean_flags(sector, level_idx, process_criticality)
    
    row = {
        'object_name': object_name,
        'sector': sector,
        'level': level,
        'service_scale': service_scale,
        'process_criticality': process_criticality,
        'num_users': num_users,
        'num_territories': num_territories,
        'predicted_financial_damage': fin_damage,
        'recovery_time_hours': rec_time,
        'critical_processes_count': crit_proc,
        'integrations_count': integr,
        'personal_data_subjects': pers_data,
        'affected_employees': aff_emp,
        'continuous_operation': continuous_op,
        'uses_automated_control_system': uses_asu,
        'provides_gov_services': gov_services,
        'life_health_impact': life_health,
        'ecological_impact': eco,
        'defense_impact': defense,
        'public_order_impact': pub_order,
        'transport_impact': transport,
        'communication_impact': comm,
        'sensitive_info': sensitive
    }
    # Целевая метка
    row['category_level'] = calculate_category(row)
    data.append(row)

# --- Сохранение ---
df = pd.DataFrame(data)
df.to_csv('synthetic_kii_data.csv', index=False, encoding='utf-8-sig')
print(f"Сгенерирован датасет из {NUM_SAMPLES} записей. Распределение по категориям:")
print(df['category_level'].value_counts().sort_index())
