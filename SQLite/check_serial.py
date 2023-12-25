# check_serial.py

import pandas as pd

# Загрузка файла CSV
csv_file_path = './reestr.csv'
df = pd.read_csv(csv_file_path)

# Поиск дубликатов в столбце 'serial'
duplicates = df[df.duplicated('serial', keep=False)]  # keep=False помечает все дубликаты

# Проверка наличия дубликатов и вывод их на экран
if not duplicates.empty:
    print("Найдены дубликаты в столбце 'serial':")
    for index, row in duplicates.iterrows():
        print(f"Порядковый номер: {index}, Строка: {row.tolist()}")
else:
    print("Дубликаты не найдены.")
