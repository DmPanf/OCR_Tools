# convert_xlsx2csv.py

import pandas as pd
from datetime import datetime

# Загрузка файла Excel
file_path = 'Reestr.xlsx'  # Укажите путь к вашему файлу .xlsx
df = pd.read_excel(file_path)

# Добавление столбцов
# Предполагается, что у вас уже есть данные для этих столбцов или вы можете их сгенерировать
df['Год'] = datetime.now().year
df['Месяц'] = datetime.now().month
df['Дата'] = datetime.now().day
df['Время'] = datetime.now().strftime('%H:%M:%S')
df['Автор'] = 'admin'  # Замените на нужное значение или переменную

# Сохранение в CSV
csv_file_path = 'reestr.csv'  # Укажите путь и имя для файла .csv
df.to_csv(csv_file_path, index=False, encoding='utf-8')
