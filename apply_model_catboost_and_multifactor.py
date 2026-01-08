# apply_model_catboost_and_multifactor.py
# ------------------------------------------------------------
# 1) Загружает csv_for_ml.csv
# 2) Загружает bankruptcy_catboost.cbm
# 3) Считает prob_bankrupt и сохраняет csv_with_prob_catboost.csv
# 4) Строит многомерную зависимость В ДИНАМИКЕ (t1/t2/t3):
#    - для каждого коэффициента: mean3, trend13, worsen_all
#    - приводит все признаки к единому направлению риска (больше=хуже)
#    - нормализует через rank(pct=True)
#    - считает интегральный risk_index_3y
# 5) Печатает таблицы для вставки в раздел и сохраняет их в CSV
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
from catboost import CatBoostClassifier, Pool


# ==========================
# Настройки
# ==========================

CSV_PATH = Path("./get_csv_data_for_ml/csv_for_ml.csv")
MODEL_PATH = Path("bankruptcy_catboost.cbm")

OUT_WITH_PROB = Path("csv_with_prob_catboost.csv")

# Итоговые таблички
OUT_RISK_DECILES = Path("tab_risk_deciles_3y.csv")
OUT_RISK_CORR = Path("tab_risk_corr_3y.csv")

TARGET_COL = "bankrupt_t"

# Базовые имена коэффициентов (суффиксы t1/t2/t3 добавляются автоматически)
BASE_COEFS = [
    "k_1200_1600",
    "k_1300_1600",
    "k_1500_1600",
    "k_14001500_1600",
    "k_neg_1300",
    "k_2400_1600",
    "k_2400_2100",
]

# Для этих коэффициентов "больше = лучше", поэтому для единого направления риска мы:
# - mean3 -> -mean3
# - trend13 -> -trend13 (чтобы ухудшение давало рост риска)
BIGGER_IS_BETTER = [
    "k_1200_1600",
    "k_1300_1600",
    "k_1500_1600",
    "k_14001500_1600",
    "k_2400_1600",
    "k_2400_2100",
]

# Какие лаги использовать (строго в рамках твоей логики "3 года до события")
LAGS = ["t1", "t2", "t3"]

# Сколько групп по risk_index (децили=10)
N_BINS = 10


# ==========================
# Утилиты
# ==========================

def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}")
    sys.exit(code)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"CSV not found: {path}")
    if path.stat().st_size == 0:
        die(f"CSV is empty: {path}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        die(f"Failed to read CSV: {path}\n{type(e).__name__}: {e}")

    if df is None or df.empty:
        die(f"CSV loaded, but dataframe is empty: {path}")

    return df


def load_catboost_model(path: Path) -> CatBoostClassifier:
    if not path.exists():
        die(f"Model file not found: {path}")
    model = CatBoostClassifier()
    try:
        model.load_model(path.as_posix())
    except Exception as e:
        die(f"Failed to load CatBoost model: {path}\n{type(e).__name__}: {e}")
    return model


def require_columns(df: pd.DataFrame, cols: list[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sample = list(df.columns)[:40]
        die(
            f"{ctx}: missing columns: {missing}\n"
            f"Hint: check column names in CSV. Example columns: {sample}"
        )


# ==========================
# Шаг 1: prob_bankrupt
# ==========================

def add_prob_bankrupt(df: pd.DataFrame, model: CatBoostClassifier) -> pd.DataFrame:
    """
    Добавляет колонку prob_bankrupt.
    Исправляет NaN в категориальных колонках (object/category) -> '__MISSING__'
    """
    if TARGET_COL not in df.columns:
        die(f"Target column not found in df: {TARGET_COL}")

    X = df.drop(columns=[TARGET_COL])

    # Категориальные колонки по dtype
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # NaN в cat -> строковый маркер
    for c in cat_cols:
        X[c] = X[c].fillna("__MISSING__").astype(str)

    pool = Pool(X, cat_features=cat_cols)

    try:
        proba = model.predict_proba(pool)[:, 1]
    except Exception as e:
        msg = (
            f"predict_proba failed: {type(e).__name__}: {e}\n\n"
            "HINT: If during training some numeric columns were passed as cat_features, "
            "you must pass the same list here. Currently detected only object/category.\n"
            f"Detected cat_cols={len(cat_cols)}: {cat_cols[:20]}{'...' if len(cat_cols) > 20 else ''}\n"
        )
        die(msg)

    out = df.copy()
    out["prob_bankrupt"] = proba
    return out


# ==========================
# Шаг 2: динамический risk_index_3y (t1/t2/t3)
# ==========================

def build_risk_index_3y(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Строит интегральный risk_index_3y по динамике коэффициентов за 3 года (t1,t2,t3).

    Для каждого базового коэффициента b:
      - b__mean3  : средний уровень за 3 года
      - b__trend13: тренд за период (t1 - t3)
      - b__worsen_all: монотонное ухудшение (3 года подряд)

    Затем:
      - приводит все признаки к единому направлению риска (больше=хуже)
      - нормализует через rank(pct=True)
      - risk_index_3y = среднее по квантильным рангам (0..1)

    Возвращает (df_with_index, список_динамических_признаков_использованных_в_индексе)
    """
    # Проверим наличие всех лагов для всех коэффициентов
    need_cols = [f"{b}_{t}" for b in BASE_COEFS for t in LAGS]
    require_columns(df, need_cols, "build_risk_index_3y")

    dyn = pd.DataFrame(index=df.index)

    for b in BASE_COEFS:
        c1, c2, c3 = f"{b}_t1", f"{b}_t2", f"{b}_t3"

        x1 = pd.to_numeric(df[c1], errors="coerce")
        x2 = pd.to_numeric(df[c2], errors="coerce")
        x3 = pd.to_numeric(df[c3], errors="coerce")

        # Уровень (среднее за 3 года)
        dyn[f"{b}__mean3"] = pd.concat([x1, x2, x3], axis=1).mean(axis=1, skipna=True)

        # Тренд (как изменилось за 3 года)
        dyn[f"{b}__trend13"] = x1 - x3

        # Ухудшение все годы подряд
        if b in BIGGER_IS_BETTER:
            # "больше=лучше" => ухудшение = падение: t1 < t2 < t3 (ухудшается по мере удаления от t1)
            dyn[f"{b}__worsen_all"] = ((x1 < x2) & (x2 < x3)).astype(int)
        else:
            # "больше=хуже" => ухудшение = рост: t1 > t2 > t3 (для риска будет наоборот, но оставим как общую идею)
            dyn[f"{b}__worsen_all"] = ((x1 > x2) & (x2 > x3)).astype(int)

    # Приведение к единому направлению риска "больше = хуже"
    # Для mean3: если "больше=лучше" -> умножаем на -1
    # Для trend13: если "больше=лучше", то отрицательный тренд = ухудшение, поэтому берём -trend13
    for b in BIGGER_IS_BETTER:
        dyn[f"{b}__mean3"] = -dyn[f"{b}__mean3"]
        dyn[f"{b}__trend13"] = -dyn[f"{b}__trend13"]

    # Нормализация каждого динамического признака через percentile-rank (0..1)
    dyn_features = list(dyn.columns)
    q_cols = []
    for c in dyn_features:
        qc = c + "__q"
        dyn[qc] = dyn[c].rank(pct=True)
        q_cols.append(qc)

    out = df.copy()
    out["risk_index_3y"] = dyn[q_cols].mean(axis=1, skipna=True)

    return out, dyn_features


# ==========================
# Шаг 3: таблицы для вставки
# ==========================

def make_tables_3y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1) Таблица децилей по risk_index_3y: n, mean_prob, bankrupt_rate, mean_risk
    2) Корреляции (Spearman) risk_index_3y с prob_bankrupt и bankrupt_t
    3) prob_bankrupt по факту bankrupt_t (describe)
    """
    need = ["risk_index_3y", "prob_bankrupt", TARGET_COL]
    require_columns(df, need, "make_tables_3y")

    d = df.dropna(subset=need).copy()
    d["risk_bin"] = pd.qcut(d["risk_index_3y"], q=N_BINS, duplicates="drop")

    tab_dec = (
        d.groupby("risk_bin", observed=True)
        .agg(
            n=("risk_index_3y", "size"),
            mean_risk=("risk_index_3y", "mean"),
            mean_prob=("prob_bankrupt", "mean"),
            bankrupt_rate=(TARGET_COL, "mean"),
        )
        .reset_index()
    )

    corr = pd.DataFrame(
        {
            "metric": [
                "spearman(risk_index_3y, prob_bankrupt)",
                "spearman(risk_index_3y, bankrupt_t)",
            ],
            "value": [
                d["risk_index_3y"].corr(d["prob_bankrupt"], method="spearman"),
                d["risk_index_3y"].corr(d[TARGET_COL], method="spearman"),
            ],
        }
    )

    by_target = (
        d.groupby(TARGET_COL)["prob_bankrupt"]
        .describe()[["count", "mean", "50%", "25%", "75%"]]
        .reset_index()
    )

    return tab_dec, corr, by_target


# ==========================
# main
# ==========================

def main() -> None:
    print("[STEP] Read CSV...")
    df = read_csv_safe(CSV_PATH)
    print(f"[OK] Loaded: rows={len(df):,} cols={df.shape[1]} | file={CSV_PATH}")
    print("[OK] Columns:", list(df.columns)[:20], "..." if df.shape[1] > 20 else "")

    if TARGET_COL not in df.columns:
        die(f"Target column '{TARGET_COL}' not found in CSV.")

    # Валидация наличия t1/t2/t3 по коэффициентам
    need_cols = [f"{b}_{t}" for b in BASE_COEFS for t in LAGS]
    require_columns(df, need_cols, "main")

    print("\n[STEP] Load CatBoost model...")
    model = load_catboost_model(MODEL_PATH)
    print(f"[OK] Model loaded: {MODEL_PATH}")

    print("\n[STEP] Predict prob_bankrupt...")
    dfp = add_prob_bankrupt(df, model)
    print("[OK] prob_bankrupt stats:")
    print(dfp["prob_bankrupt"].describe())

    print("\n[STEP] Save CSV with probabilities...")
    dfp.to_csv(OUT_WITH_PROB, index=False)
    print(f"[OK] Saved: {OUT_WITH_PROB}")

    print("\n[STEP] Build dynamic multi-factor risk_index_3y (t1/t2/t3)...")
    dfp2, dyn_features = build_risk_index_3y(dfp)
    print("[OK] risk_index_3y stats:")
    print(dfp2["risk_index_3y"].describe())
    print(f"[OK] Dynamic features used: {len(dyn_features)}")
    print("     Example:", dyn_features[:10], "..." if len(dyn_features) > 10 else "")

    print("\n[STEP] Build tables...")
    tab_dec, corr, by_target = make_tables_3y(dfp2)

    print("\n=== TABLE: risk deciles (3-year profile) ===")
    print(tab_dec.to_string(index=False))

    print("\n=== TABLE: correlations (sanity check) ===")
    print(corr.to_string(index=False))

    print("\n=== TABLE: prob_bankrupt by bankrupt_t ===")
    print(by_target.to_string(index=False))

    tab_dec.to_csv(OUT_RISK_DECILES, index=False)
    corr.to_csv(OUT_RISK_CORR, index=False)
    print(f"\n[OK] Saved tables: {OUT_RISK_DECILES}, {OUT_RISK_CORR}")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
