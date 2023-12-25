# make_sqlite.py
 
import sqlite3
import pandas as pd

# Создание базы данных и таблицы
conn = sqlite3.connect('reestr.db')
create_table_query = '''
CREATE TABLE IF NOT EXISTS measurements (
    serial TEXT,
    type TEXT,
    value REAL,
    image TEXT,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    time TEXT,
    user TEXT
)
'''
conn.execute(create_table_query)
conn.commit()

# Чтение данных из файла CSV
df = pd.read_csv('reestr.csv')

# Предварительная обработка данных, если это необходимо

# Вставка данных в базу данных
df.to_sql('measurements', conn, if_exists='append', index=False)

# Закрытие соединения с базой данных
conn.close()
