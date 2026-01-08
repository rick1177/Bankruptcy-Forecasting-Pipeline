"""
train_lightgbm.py — обучение модели банкротства на LightGBM (GBDT)

Логика как в train_model.py (CatBoost):
1) Контроль окружения
2) Чтение CSV
3) Разбиение по dataset_split
4) Формирование X/y, удаление утечек
5) Категориальные признаки:
   - Основной режим: dtype='category' + categorical_feature
   - Fallback: One-Hot Encoding
6) Обучение LGBMClassifier
7) Метрики: AUC, confusion matrix, classification report
8) Сохранение модели в .txt (booster)
9) Тайминги

Зависимости:
pip install lightgbm scikit-learn pandas
"""

import sys
import time

import pandas as pd

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "Не найден пакет lightgbm. Установи: pip install lightgbm"
    ) from e


print("[PYTHON]", sys.executable)
t0_all = time.perf_counter()

DATA_PATH = r"D:\Code\new_bfo\get_csv_data_for_ml\csv_for_ml.csv"

t0 = time.perf_counter()
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"[TIME] read_csv: {time.perf_counter() - t0:.2f}s | rows={len(df):,} cols={df.shape[1]:,}")

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

drop_cols = [
    "inn",
    "base_year",
    "dataset_split",
    "bankrupt_t",
    "active_t",  # утечка
]
drop_cols = [c for c in drop_cols if c in df.columns]

X_train_raw = train_df.drop(columns=drop_cols)
y_train = train_df["bankrupt_t"].astype(int)

X_test_raw = test_df.drop(columns=drop_cols)
y_test = test_df["bankrupt_t"].astype(int)

# ---- утилита: оценить дисбаланс ----
pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

# ====== 1) Основной режим: categorical_feature ======
def train_with_categorical():
    t0 = time.perf_counter()

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    # Категориальные: object -> category, NaN -> отдельная категория "__MISSING__"
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

    for c in cat_cols:
        X_train[c] = X_train[c].fillna("__MISSING__").astype(str).astype("category")
        X_test[c] = X_test[c].fillna("__MISSING__").astype(str).astype("category")

        # Важно: синхронизируем категории train/test,
        # иначе LightGBM иногда ругается на разные наборы.
        X_test[c] = X_test[c].cat.set_categories(X_train[c].cat.categories)

    print(f"[INFO] categorical mode | features={X_train.shape[1]} | cat={len(cat_cols)}")
    print(f"[TIME] prep categorical: {time.perf_counter() - t0:.2f}s")

    model = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        objective="binary",
        metric="auc",
        scale_pos_weight=scale_pos_weight,
    )

    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(period=50)],
        categorical_feature=cat_cols if cat_cols else "auto",
    )
    print(f"[TIME] fit: {time.perf_counter() - t0:.2f}s")

    return model, X_test


# ====== 2) Fallback: One-Hot ======
def train_with_onehot():
    t0 = time.perf_counter()

    all_raw = pd.concat([X_train_raw, X_test_raw], axis=0)
    obj_cols = [c for c in all_raw.columns if all_raw[c].dtype == "object"]
    for c in obj_cols:
        all_raw[c] = all_raw[c].fillna("__MISSING__").astype(str)

    all_ohe = pd.get_dummies(all_raw, columns=obj_cols, dummy_na=False)

    X_train = all_ohe.iloc[: len(X_train_raw)].copy()
    X_test = all_ohe.iloc[len(X_train_raw) :].copy()

    print(f"[INFO] one-hot mode | features={X_train.shape[1]:,}")
    print(f"[TIME] one-hot: {time.perf_counter() - t0:.2f}s")

    model = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        objective="binary",
        metric="auc",
        scale_pos_weight=scale_pos_weight,
    )

    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(period=50)],
    )
    print(f"[TIME] fit: {time.perf_counter() - t0:.2f}s")

    return model, X_test


# ====== Запуск: сначала categorical, если упадёт -> one-hot ======
try:
    model, X_test_used = train_with_categorical()
except Exception as e:
    print("\n[WARN] categorical mode failed, fallback to one-hot.")
    print("[WARN] error:", repr(e))
    model, X_test_used = train_with_onehot()

# ====== Оценка ======
t0 = time.perf_counter()
proba_test = model.predict_proba(X_test_used)[:, 1]
auc = roc_auc_score(y_test, proba_test)
print("\nTEST AUC:", round(auc, 4))
print(f"[TIME] predict+auc: {time.perf_counter() - t0:.2f}s")

threshold = 0.2
pred_test = (proba_test >= threshold).astype(int)

print("\nConfusion matrix (threshold =", threshold, "):")
print(confusion_matrix(y_test, pred_test))

print("\nClassification report:")
print(classification_report(y_test, pred_test, digits=4))

# ====== Сохранение ======
t0 = time.perf_counter()
MODEL_PATH = "bankruptcy_lightgbm.txt"
model.booster_.save_model(MODEL_PATH)
print("\nSaved model to:", MODEL_PATH)
print(f"[TIME] save_model: {time.perf_counter() - t0:.2f}s")

print(f"\n[TIME] TOTAL: {time.perf_counter() - t0_all:.2f}s")
