# build_visuals_3_5.py
# ------------------------------------------------------------
# Строит визуализации для раздела 3.5 по агрегатам из 3.4:
#   - decile-графики по tab_risk_deciles_3y.csv
#   - комбинированный график (prob vs bankrupt_rate)
#   - зональный анализ (low/medium/high) из децилей (взвешенно по n)
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Настройки путей
# =========================

DECILES_PATH = Path("tab_risk_deciles_3y.csv")  # вход: из твоего скрипта 3.4
OUT_DIR = Path("out_3_5")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Утилиты
# =========================

def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}")
    sys.exit(code)


def read_deciles(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"File not found: {path.resolve()}")

    df = pd.read_csv(path)

    need = {"risk_bin", "n", "mean_risk", "mean_prob", "bankrupt_rate"}
    miss = need - set(df.columns)
    if miss:
        die(f"Missing columns in {path.name}: {sorted(miss)}. "
            f"Columns found: {list(df.columns)}")

    # На всякий случай приводим типы
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["mean_risk"] = pd.to_numeric(df["mean_risk"], errors="coerce")
    df["mean_prob"] = pd.to_numeric(df["mean_prob"], errors="coerce")
    df["bankrupt_rate"] = pd.to_numeric(df["bankrupt_rate"], errors="coerce")

    if df[["n", "mean_risk", "mean_prob", "bankrupt_rate"]].isna().any().any():
        # Не стопим, но предупреждаем
        print("[WARN] Some numeric columns have NaN after type conversion. "
              "Check input file integrity.")

    # Восстановим границы интервала из risk_bin:
    # пример: "(0.253, 0.434]" или "(0.57, 0.831]"
    def parse_bin_edges(s: str) -> tuple[float, float]:
        s = str(s).strip()
        m = re.match(r"^[\(\[]\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*[\)\]]$", s)
        if not m:
            # Иногда pandas может записать иначе — пробуем вытащить числа
            nums = re.findall(r"[0-9]+\.[0-9]+|[0-9]+", s)
            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])
            return np.nan, np.nan
        return float(m.group(1)), float(m.group(2))

    edges = df["risk_bin"].apply(parse_bin_edges)
    df["risk_left"] = [a for a, _ in edges]
    df["risk_right"] = [b for _, b in edges]

    # Отсортируем по левой границе (это и будет порядок децилей)
    df = df.sort_values(["risk_left", "risk_right"], ascending=True).reset_index(drop=True)
    df["decile"] = np.arange(1, len(df) + 1)

    return df


def save_plot(fig, filename: str) -> None:
    out = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved plot: {out}")


def make_decile_bar(df: pd.DataFrame, y_col: str, y_label: str, title: str, filename: str) -> None:
    x = df["decile"].astype(int)
    y = df[y_col].astype(float)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, y)
    ax.set_xlabel("Номер дециля риск-профиля (D1 … D10)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)

    # чуть информативнее: подписи значений
    for xi, yi in zip(x, y):
        ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)

    save_plot(fig, filename)


def make_combo(df: pd.DataFrame, filename: str) -> None:
    x = df["decile"].astype(int)
    bankrupt_rate = df["bankrupt_rate"].astype(float)
    mean_prob = df["mean_prob"].astype(float)

    fig, ax1 = plt.subplots(figsize=(9, 4.8))

    # Столбцы: фактическая доля банкротов
    ax1.bar(x, bankrupt_rate)
    ax1.set_xlabel("Номер дециля риск-профиля (D1 … D10)")
    ax1.set_ylabel("Фактическая доля банкротов (bankrupt_rate)")
    ax1.set_xticks(x)

    # Линия: средняя прогнозная вероятность
    ax2 = ax1.twinx()
    ax2.plot(x, mean_prob, marker="o")
    ax2.set_ylabel("Средняя прогнозируемая вероятность банкротства (prob_bankrupt_mean)")

    ax1.set_title("Сопоставление прогнозной вероятности и фактической доли банкротств по децилям")

    save_plot(fig, filename)


def build_zones_from_deciles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строим зоны риска как укрупнение децилей по mean_risk (взвешенно по n).

    По твоей таблице зон (пример):
      Low:    0.25–0.45
      Medium: 0.45–0.53
      High:   0.53–0.83

    Мы повторяем эту логику, но не «вручную», а на основе mean_risk децилей.
    """

    # границы зон (можешь поменять, если решишь иначе)
    low_max = 0.45
    mid_max = 0.53

    d = df.copy()

    def zone(mr: float) -> str:
        if pd.isna(mr):
            return "Не определено"
        if mr <= low_max:
            return "Низкий риск"
        if mr <= mid_max:
            return "Средний риск"
        return "Высокий риск"

    d["risk_zone"] = d["mean_risk"].apply(zone)

    # Взвешенные средние по n (важно!)
    def wavg(group: pd.DataFrame, col: str) -> float:
        w = group["n"].astype(float).values
        x = group[col].astype(float).values
        if np.sum(w) == 0:
            return float("nan")
        return float(np.sum(w * x) / np.sum(w))

    agg = (
        d.groupby("risk_zone", as_index=False)
        .apply(lambda g: pd.Series({
            "n": int(g["n"].sum()),
            "risk_index_mean": wavg(g, "mean_risk"),
            "prob_bankrupt_mean": wavg(g, "mean_prob"),
            "bankrupt_rate": wavg(g, "bankrupt_rate"),
            # диапазон индекса в зоне — по границам интервалов risk_bin
            "risk_index_min": float(np.nanmin(g["risk_left"].values)),
            "risk_index_max": float(np.nanmax(g["risk_right"].values)),
        }))
        .reset_index(drop=True)
    )

    # Приведём порядок зон
    order = {"Низкий риск": 1, "Средний риск": 2, "Высокий риск": 3, "Не определено": 99}
    agg["__ord"] = agg["risk_zone"].map(order).fillna(99).astype(int)
    agg = agg.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

    # Красивый диапазон строкой
    agg["risk_index_range"] = agg.apply(
        lambda r: f"{r['risk_index_min']:.2f}–{r['risk_index_max']:.2f}", axis=1
    )

    # Оставим нужные столбцы (как у тебя в тексте)
    out = agg[[
        "risk_zone",
        "risk_index_range",
        "risk_index_mean",
        "prob_bankrupt_mean",
        "bankrupt_rate",
        "n",
    ]].copy()

    return out


def make_zone_bars(z: pd.DataFrame, filename: str) -> None:
    x = z["risk_zone"].astype(str)
    prob = z["prob_bankrupt_mean"].astype(float)
    fact = z["bankrupt_rate"].astype(float)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    # сделаем два ряда столбцов
    idx = np.arange(len(x))
    width = 0.38

    ax.bar(idx - width/2, prob, width, label="Средняя прогнозная вероятность (prob_bankrupt_mean)")
    ax.bar(idx + width/2, fact, width, label="Фактическая доля банкротов (bankrupt_rate)")

    ax.set_xticks(idx)
    ax.set_xticklabels(x)
    ax.set_ylabel("Значение показателя")
    ax.set_title("Зональный анализ: прогнозная вероятность и фактическая доля банкротов")
    ax.legend()

    # подписи
    for i, (p, f) in enumerate(zip(prob, fact)):
        ax.text(i - width/2, p, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, f, f"{f:.3f}", ha="center", va="bottom", fontsize=8)

    save_plot(fig, filename)


# =========================
# main
# =========================

def main() -> None:
    print("[STEP] Read deciles table...")
    dec = read_deciles(DECILES_PATH)
    print("[OK] Loaded:", DECILES_PATH.resolve())
    print(dec[["decile", "risk_bin", "n", "mean_risk", "mean_prob", "bankrupt_rate"]].to_string(index=False))

    # 3.5.1
    make_decile_bar(
        dec,
        y_col="mean_prob",
        y_label="Средняя прогнозируемая вероятность банкротства (prob_bankrupt_mean)",
        title="Зависимость средней прогнозной вероятности банкротства от дециля риск-профиля",
        filename="3_5_1_prob_by_decile.png",
    )

    # 3.5.2
    make_decile_bar(
        dec,
        y_col="bankrupt_rate",
        y_label="Фактическая доля банкротов (bankrupt_rate)",
        title="Фактическая доля банкротов по децилям риск-профиля",
        filename="3_5_2_bankrupt_rate_by_decile.png",
    )

    # 3.5.3
    make_combo(dec, filename="3_5_3_combo_prob_vs_fact.png")

    # 3.5.4 зоны
    print("[STEP] Build risk zones from deciles (weighted by n)...")
    zones = build_zones_from_deciles(dec)
    zones_out = OUT_DIR / "tab_risk_zones_3y.csv"
    zones.to_csv(zones_out, index=False, encoding="utf-8-sig")
    print("[OK] Saved zones table:", zones_out)
    print(zones.to_string(index=False))
    

    make_zone_bars(zones, filename="3_5_4_zones_prob_vs_fact.png")

    print("\n[OK] Done. See outputs in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
