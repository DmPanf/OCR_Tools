# del_Unnamed.py

import pandas as pd

# Загрузка файла CSV
csv_file_path = './reestr.csv'  # Укажите путь к вашему файлу .csv
df = pd.read_csv(csv_file_path)

# Удаление столбца 'Unnamed', если он есть
# Столбец может иметь название вроде 'Unnamed: 4'
if 'Unnamed' in df.columns:
    df.drop('Unnamed', axis=1, inplace=True)

# Сохранение изменений в файл CSV
df.to_csv(csv_file_path, index=False, encoding='utf-8')
