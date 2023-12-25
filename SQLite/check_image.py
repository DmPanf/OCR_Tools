# check_image.py
import pandas as pd

# Загрузка файла CSV
csv_file_path = './reestr.csv'
df = pd.read_csv(csv_file_path)

# Поиск дубликатов в столбце 'serial'
duplicates = df[df.duplicated('image', keep=False)]  # keep=False помечает все дубликаты

# Проверка наличия дубликатов и вывод их на экран
if not duplicates.empty:
    print("Найдены дубликаты в столбце 'image':")
    for index, row in duplicates.iterrows():
        print(f"Порядковый номер: {index}, Строка: {row.tolist()}")
else:
    print("Дубликаты не найдены.")
