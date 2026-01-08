from pathlib import Path
import duckdb
import time

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "local.duckdb"
SQL_QUERY_PATH = BASE_DIR / "sql_query.sql"
EXPORT_SQL_PATH = BASE_DIR / "export.sql"


def main() -> None:
    con = duckdb.connect(DB_PATH.as_posix())

    t0 = time.perf_counter()
    con.execute(SQL_QUERY_PATH.read_text(encoding="utf-8"))  # готовит VIEW/CTE окружение
    t1 = time.perf_counter()


    export_sql = EXPORT_SQL_PATH.read_text(encoding="utf-8")
    out_path = (BASE_DIR / "csv_for_ml.csv").as_posix()
    export_sql = export_sql.replace("__OUT_PATH__", out_path)
    con.execute(export_sql)
    t2 = time.perf_counter()

    con.close()

    # Размер экспортного файла (ожидаем, что export.sql пишет csv_for_ml.csv рядом со скриптом)
    out_path = BASE_DIR / "csv_for_ml.csv"
    if out_path.exists():
        size_bytes = out_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"file:   {out_path.name} ({size_mb:.2f} MB)")
    else:
        print(f"file:   {out_path.name} (not found)")

    print(f"prep:   {t1 - t0:.2f}s")
    print(f"export: {t2 - t1:.2f}s")
    print(f"total:  {t2 - t0:.2f}s")


if __name__ == "__main__":
    main()
