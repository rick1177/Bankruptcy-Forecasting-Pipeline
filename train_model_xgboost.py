"""
train_xgboost.py — обучение модели банкротства на XGBoost (GBDT)

Логика как в train_model.py (CatBoost):
1) Контроль окружения (sys.executable)
2) Чтение CSV итоговой таблицы
3) Разбиение по dataset_split (train/test)
4) Формирование X/y, удаление служебных полей и утечек
5) Обработка категориальных признаков через One-Hot Encoding (pd.get_dummies)
6) Обучение XGBClassifier
7) Метрики: AUC, confusion matrix, classification report
8) Сохранение модели в .json (нативный формат XGBoost)
9) Тайминги

Зависимости:
pip install xgboost scikit-learn pandas
"""

import sys
import time

import pandas as pd

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "Не найден пакет xgboost. Установи: pip install xgboost"
    ) from e


print("[PYTHON]", sys.executable)
t0_all = time.perf_counter()

# ====== 1) Загрузка данных ======
DATA_PATH = r"D:\Code\new_bfo\get_csv_data_for_ml\csv_for_ml.csv"

t0 = time.perf_counter()
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"[TIME] read_csv: {time.perf_counter() - t0:.2f}s | rows={len(df):,} cols={df.shape[1]:,}")

# ====== 2) Проверки и разбиение ======
required_cols = {"bankrupt_t", "dataset_split"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"В данных нет обязательных колонок: {missing}")

train_df = df[df["dataset_split"] == "train"].copy()
test_df = df[df["dataset_split"] == "test"].copy()

print("Rows:", len(df), "train:", len(train_df), "test:", len(test_df))
print("Target mean (overall):", df["bankrupt_t"].mean())
print("Target mean (train):", train_df["bankrupt_t"].mean())
print("Target mean (test):", test_df["bankrupt_t"].mean())

# ====== 3) Выбор фичей ======
drop_cols = [
    "inn",
    "base_year",
    "dataset_split",
    "bankrupt_t",
    "active_t",  # утечка (год t)
]
drop_cols = [c for c in drop_cols if c in df.columns]

X_train_raw = train_df.drop(columns=drop_cols)
y_train = train_df["bankrupt_t"].astype(int)

X_test_raw = test_df.drop(columns=drop_cols)
y_test = test_df["bankrupt_t"].astype(int)

# ====== 4) Категориальные -> One-Hot ======
# XGBoost надёжнее всего обучать на числах => one-hot всех object.
t0 = time.perf_counter()

# Склеиваем train+test, чтобы после OHE совпали наборы колонок
all_raw = pd.concat([X_train_raw, X_test_raw], axis=0)

# Для object — заполним пропуски и приведём к строке
obj_cols = [c for c in all_raw.columns if all_raw[c].dtype == "object"]
for c in obj_cols:
    all_raw[c] = all_raw[c].fillna("__MISSING__").astype(str)

# Для числовых — NaN оставляем как есть (XGBoost умеет missing)
all_ohe = pd.get_dummies(all_raw, columns=obj_cols, dummy_na=False)

# Разделяем обратно
X_train = all_ohe.iloc[: len(X_train_raw)].copy()
X_test = all_ohe.iloc[len(X_train_raw) :].copy()

print(f"[TIME] one-hot: {time.perf_counter() - t0:.2f}s | features={X_train.shape[1]:,}")

# ====== 5) Модель ======
# scale_pos_weight можно оценить как neg/pos, но у тебя уже было balanced в CatBoost.
# Здесь используем scale_pos_weight как базовую компенсацию дисбаланса.
pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

model = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    random_state=42,
    n_jobs=-1,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
)

# ====== 6) Обучение ======
t0 = time.perf_counter()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)
print(f"[TIME] fit: {time.perf_counter() - t0:.2f}s")

# ====== 7) Оценка ======
t0 = time.perf_counter()
proba_test = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba_test)
print("\nTEST AUC:", round(auc, 4))
print(f"[TIME] predict+auc: {time.perf_counter() - t0:.2f}s")

threshold = 0.2
pred_test = (proba_test >= threshold).astype(int)

print("\nConfusion matrix (threshold =", threshold, "):")
print(confusion_matrix(y_test, pred_test))

print("\nClassification report:")
print(classification_report(y_test, pred_test, digits=4))

# ====== 8) Сохранение модели ======
t0 = time.perf_counter()
MODEL_PATH = "bankruptcy_xgboost.json"
model.get_booster().save_model(MODEL_PATH)   # <-- работает стабильнее
print("\nSaved model to:", MODEL_PATH)
print(f"[TIME] save_model: {time.perf_counter() - t0:.2f}s")
print(f"\n[TIME] TOTAL: {time.perf_counter() - t0_all:.2f}s")
