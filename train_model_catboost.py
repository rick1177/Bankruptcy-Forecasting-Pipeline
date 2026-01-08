"""
train_model_catboost.py — обучение модели банкротства на CatBoost

Что делает скрипт:
1) Печатает путь интерпретатора Python (чтобы видеть, что запустили именно venv).
2) Загружает CSV с итоговой таблицей для ML.
3) Делит на train/test по колонке dataset_split.
4) Готовит X/y, выкидывает служебные поля и утечки.
5) Исправляет категориальные признаки (cat_features): заполняет NaN и приводит к строке,
   иначе CatBoost падает с ошибкой "bad object for id: nan".
6) Обучает CatBoostClassifier с логом прогресса (verbose) — CatBoost показывает ETA/remaining time.
7) Считает AUC, confusion matrix и classification report.
8) Сохраняет модель в .cbm.
9) Печатает тайминги ключевых этапов и общий тайминг.

Важно про таблицу:
- Признаки *_t1, *_t2, *_t3 оставляем (это прошлые годы/лаги).
- Поля целевого года t (например active_t) выкидываем, иначе утечка.
"""

import sys
import time

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


# ========== 0) Контроль окружения ==========
# Печатаем, каким интерпретатором реально запускается файл.
# Если тут НЕ путь вида ...\.venv\Scripts\python.exe — значит VS Code запускает не то окружение.
print("[PYTHON]", sys.executable)

t0_all = time.perf_counter()


# ========== 1) Загрузка данных ==========
# Важно:
# - r"..." для Windows-пути, чтобы \ не интерпретировались как спец-символы.
# - low_memory=False: pandas читает файл более "цельно" и меньше делает смешанных типов,
#   что уменьшает вероятность DtypeWarning и неожиданных dtype=object.
DATA_PATH = r"D:\Code\new_bfo\get_csv_data_for_ml\csv_for_ml.csv"

t0 = time.perf_counter()
df = pd.read_csv(DATA_PATH, low_memory=False)
print(
    f"[TIME] read_csv: {time.perf_counter() - t0:.2f}s | "
    f"rows={len(df):,} cols={df.shape[1]:,}"
)


# ========== 2) Проверки и разбиение ==========
# Минимально необходимые поля (без них модель не собрать):
# - bankrupt_t: таргет (0/1)
# - dataset_split: разбиение на train/test
required_cols = {"bankrupt_t", "dataset_split"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"В данных нет обязательных колонок: {missing}")

# Разделяем строго по готовой разметке (как ты делал в итоговой таблице для ML).
# .copy() — чтобы дальше безопасно менять данные (pandas иногда ругается на chained assignment).
train_df = df[df["dataset_split"] == "train"].copy()
test_df = df[df["dataset_split"] == "test"].copy()

# Базовые sanity-checks: размеры и доля положительного класса (банкротств).
print("Rows:", len(df), "train:", len(train_df), "test:", len(test_df))
print("Target mean (overall):", df["bankrupt_t"].mean())
print("Target mean (train):", train_df["bankrupt_t"].mean())
print("Target mean (test):", test_df["bankrupt_t"].mean())


# ========== 3) Выбор фичей (X) и таргета (y) ==========
# drop_cols — это то, что НЕ должно попасть в признаки:
# 1) Идентификаторы и служебные:
#    - inn: идентификатор (в обучение не нужно, иначе будет "запоминание" компаний)
#    - base_year: год базового среза (как правило, тоже не нужен как фича для первой версии)
#    - dataset_split: служебное разбиение
# 2) Таргет:
#    - bankrupt_t: то, что предсказываем
# 3) Утечка:
#    - active_t: факт активности в целевом году t (это уже "знание будущего")
drop_cols = [
    "inn",
    "base_year",
    "dataset_split",
    "bankrupt_t",
    "active_t",  # утечка (год t)
]

# На случай если какой-то колонки вдруг нет в файле — убираем отсутствующие,
# чтобы .drop() не падал.
drop_cols = [c for c in drop_cols if c in df.columns]

# Формируем X/y:
# X_*: все признаки (включая *_t1, *_t2, *_t3 и тренды/дельты)
# y_*: bankrupt_t (0/1)
X_train = train_df.drop(columns=drop_cols)
y_train = train_df["bankrupt_t"].astype(int)

X_test = test_df.drop(columns=drop_cols)
y_test = test_df["bankrupt_t"].astype(int)


# ========== 3.1) Категориальные признаки и исправление NaN ==========
# В твоём запуске CatBoost упал с:
# "Invalid type for cat_feature ... = nan : cat_features must be integer or string"
#
# Причина:
# - Мы определяли категориальные признаки как dtype == "object"
# - В таких колонках встречаются NaN (пустые значения)
# - CatBoost для cat_features не принимает NaN как значение категории.
#
# Решение:
# - Находим все object-колонки -> считаем их категориальными
# - Заполняем NaN специальным токеном "__MISSING__"
# - Приводим значения к строке (astype(str)), чтобы тип был строго string
#   (CatBoost принимает string/int для категорий)
cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

# (опционально) можно вывести, какие именно cat-колонки содержали пропуски:
cat_cols_with_na = [c for c in cat_cols if X_train[c].isna().any()]
if cat_cols_with_na:
    print("[INFO] Cat cols with NA:", cat_cols_with_na[:20], "..." if len(cat_cols_with_na) > 20 else "")

for c in cat_cols:
    X_train[c] = X_train[c].fillna("__MISSING__").astype(str)
    X_test[c] = X_test[c].fillna("__MISSING__").astype(str)

# CatBoost Pool принимает cat_features как:
# - список индексов колонок (0..n-1), либо
# - список имён колонок (в новых версиях часто тоже можно, но индексы надёжнее).
cat_features = [X_train.columns.get_loc(c) for c in cat_cols]

# Создаём Pool — оптимальный контейнер данных для CatBoost.
# Он хранит данные, метки и информацию о категориальных признаках.
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

print(f"Features: {X_train.shape[1]} | Cat features: {len(cat_features)}")


# ========== 4) Модель ==========
# Параметры:
# - iterations: максимальное число итераций (деревьев в бустинге)
# - learning_rate: шаг обучения
# - depth: глубина дерева
# - auto_class_weights="Balanced": балансировка классов (важно при редких банкротствах)
# - eval_metric="AUC": ключевая метрика качества
# - verbose + metric_period: как часто печатать прогресс.
#   В этом прогрессе CatBoost обычно показывает ETA / remaining time.
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5,
    random_seed=42,
    auto_class_weights="Balanced",
    verbose=50,       # чаще печатает прогресс -> лучше видно "сколько осталось"
    metric_period=50  # синхронизируем частоту печати метрик
)

# Обучение:
# - eval_set=test_pool: по нему считаются метрики и выбирается best_model
# - use_best_model=True: сохранит лучшую итерацию по eval_metric
t0 = time.perf_counter()
model.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True
)
print(f"[TIME] fit: {time.perf_counter() - t0:.2f}s")


# ========== 5) Оценка качества ==========
# predict_proba -> вероятность класса 1 (банкротство)
t0 = time.perf_counter()
proba_test = model.predict_proba(test_pool)[:, 1]

# AUC — "ранжирующая" метрика: насколько хорошо модель ставит банкротства выше небанкротств.
auc = roc_auc_score(y_test, proba_test)
print("\nTEST AUC:", round(auc, 4))
print(f"[TIME] predict+auc: {time.perf_counter() - t0:.2f}s")

# Дальше: превращаем вероятности в 0/1 по порогу.
# 0.5 для дисбаланса почти всегда плохо, поэтому ставим 0.2 как стартовую гипотезу.
threshold = 0.2
pred_test = (proba_test >= threshold).astype(int)

print("\nConfusion matrix (threshold =", threshold, "):")
print(confusion_matrix(y_test, pred_test))

print("\nClassification report:")
print(classification_report(y_test, pred_test, digits=4))


# ========== 6) Сохранение модели ==========
t0 = time.perf_counter()
MODEL_PATH = "bankruptcy_catboost.cbm"
model.save_model(MODEL_PATH)
print("\nSaved model to:", MODEL_PATH)
print(f"[TIME] save_model: {time.perf_counter() - t0:.2f}s")

print(f"\n[TIME] TOTAL: {time.perf_counter() - t0_all:.2f}s")
