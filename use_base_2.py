import polars as pl
import json
import duckdb
import re
from datetime import datetime

# Загрузка file_map из файла
with open("parquet_map.json", "r", encoding="utf-8") as f:
    report = json.load(f)
    file_map = report["files"]  # Извлекаем file_map из отчёта

# Функция для извлечения года из пути файла
def extract_year(file_path):
    match = re.search(r"year=(\d{4})", file_path)  # Ищем 'year=' и 4 цифры после него
    return int(match.group(1)) if match else None


# Список путей к файлам за все годы
file_paths = [info["path"] for info in file_map.values() if info.get("path")]

# Создаём ленивые DataFrame с правильным добавлением года
dfs = []
for path in file_paths:
    year = extract_year(path)
    df = pl.scan_parquet(path).with_columns(pl.lit(year).cast(pl.Int32).alias("year"))
    dfs.append(df)

# Подключение к DuckDB
conn = duckdb.connect()

# SQL-запрос для объединения данных через DuckDB
start_time = datetime.now()
print(f'Начало выполнения запроса с применением DuckDB (время: {start_time.strftime("%H:%M:%S")})')

query = f"""
SELECT year, line_1700
FROM read_parquet({file_paths})
WHERE inn = '5905000214'
"""

result = conn.execute(query).fetchdf()
end_time = datetime.now()
execution_time = (end_time - start_time).total_seconds()

print(f"Данные по предприятию 5905000214 (время сбора данных с применением DuckDB: {execution_time:.2f} сек):")
print(result)

# Объединение всех файлов в один DataFrame
combined = pl.concat(dfs)

# Фильтрация данных с использованием Polars
start_time = datetime.now()
print(f'Начало выполнения запроса с применением Polars (время: {start_time.strftime("%H:%M:%S")})')

result = (
    combined.filter(pl.col("inn") == '5905000214')
    .select(["year", "line_1700"])
    .collect()
)

end_time = datetime.now()
execution_time = (end_time - start_time).total_seconds()

print(f"Данные по предприятию 5905000214 (время сбора данных с применением Polars: {execution_time:.2f} сек):")
print(result.to_pandas())


# Вариант 1 запрос на все таблицы по годам (работает)
# Генерируем SQL UNION ALL, где каждому файлу явно присваиваем year
union_queries = []
for path in file_paths:
    year = extract_year(path)
    union_queries.append(f"SELECT *, {year} AS year FROM read_parquet('{path}')")

query = f"""
SELECT 
    'Сумма выручки за 2022 и 2023 годы по региону 59' AS "Наименование",
    SUM(line_2100) AS "Сумма"
FROM (
    {" UNION ALL ".join(union_queries)}
)
WHERE year IN (2022, 2023) AND region_taxcode = '59'
"""

result = conn.execute(query).fetchdf()
print(result)

# Вариант 2 запрос на все таблицы по годам (работает)
# Генерируем SQL UNION ALL, где каждому файлу явно присваиваем year
file_paths_str = ", ".join(f"'{p}'" for p in file_paths)

query = f"""
SELECT 
    'Сумма выручки за 2022 и 2023 годы по региону 59' AS "Наименование",
    SUM(line_2100) AS "Сумма"
FROM (
    SELECT *, 
           CAST(REGEXP_EXTRACT(filename, 'year=(\\d{{4}})', 1) AS INT) AS year 
    FROM read_parquet([{file_paths_str}], FILENAME=TRUE)
)
WHERE year IN (2022, 2023) AND region_taxcode = '59'
"""

result = conn.execute(query).fetchdf()
print(result)


# Вариант 3 запрос на все таблицы по годам (работает)
query = f"""
SELECT 
    'Сумма выручки за 2022 и 2023 годы по региону 59' AS "Наименование",
    SUM(line_2100) AS "Сумма"
FROM read_parquet({file_paths})
WHERE year IN ('2022', '2023') AND region_taxcode = '59'
"""
result = conn.execute(query).fetchdf()
print(result)

#Варинт 4 запроса на все таблицы с применением polars
start_time = datetime.now()
print(f'Начало выполнения запроса с применением Polars (время: {start_time.strftime("%H:%M:%S")})')

result = (
    combined
    .filter(pl.col("year").is_in([2022, 2023]) & (pl.col("region_taxcode") == '59'))
    .with_columns(pl.col("line_2100").cast(pl.Float64))  # Приведение типа
    .select(pl.sum("line_2100").alias("Сумма выручки за 2022 и 2023 годы по региону 59"))
    .collect()
)

end_time = datetime.now()
execution_time = (end_time - start_time).total_seconds()

print(f"Данные по запросу (время сбора данных с применением Polars: {execution_time:.2f} сек):")
print(result.to_pandas())
