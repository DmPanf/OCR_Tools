# sqlite_example.py 

import sqlite3
import random

# Подключение к базе данных
conn = sqlite3.connect('reestr.db')
cursor = conn.cursor()

# 1. Всего записей
cursor.execute("SELECT COUNT(*) FROM measurements")
total_records = cursor.fetchone()[0]
print(f"Всего записей: {total_records}")

# 2. Записей Водоснабжение
cursor.execute("SELECT COUNT(*) FROM measurements WHERE type = 'Водоснабжение'")
water_supply_records = cursor.fetchone()[0]
print(f"Записей Водоснабжение: {water_supply_records}")

# 3. Записей Электроснабжение
cursor.execute("SELECT COUNT(*) FROM measurements WHERE type = 'Электроснабжение'")
electric_supply_records = cursor.fetchone()[0]
print(f"Записей Электроснабжение: {electric_supply_records}")

# 4. Серийный номер с максимальным количеством записей
cursor.execute("SELECT serial, COUNT(*) as count FROM measurements GROUP BY serial ORDER BY count DESC LIMIT 1")
most_common_serial = cursor.fetchone()
print(f"Серийный номер с максимальным количеством записей: {most_common_serial[0]} (записей: {most_common_serial[1]})")

# 5. Вывод всей информации о случайной записе
cursor.execute("SELECT * FROM measurements")
random_record = random.choice(cursor.fetchall())
print("Информация о случайной записи:", random_record)

# Закрытие соединения с базой данных
conn.close()
