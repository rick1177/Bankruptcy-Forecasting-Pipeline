from huggingface_hub import list_repo_files, hf_hub_download, HfApi
import json
import os
import time
from datetime import datetime
import hashlib
from typing import Dict, Optional

# Конфигурация
repo_id = "irlspbru/RFSD"
cache_dir = os.path.abspath("data_cache")
os.makedirs(cache_dir, exist_ok=True)

# Получаем список файлов из репозитория
files = list_repo_files(repo_id, repo_type="dataset")
parquet_files = [f for f in files if f.endswith('.parquet')]

file_map = {}


def get_file_metadata(filename: str) -> Dict:
    """Получаем метаданные файла через API"""
    api = HfApi()
    file_info = api.get_paths_info(repo_id, [filename], repo_type="dataset")[0]
    return {
        "etag": file_info.lfs.get("oid") if file_info.lfs else None,  # Используем LFS OID как ETag
        "size": file_info.size
    }


def safe_download(filename: str, retries: int = 3) -> Dict:
    """Безопасная загрузка с проверкой целостности"""
    result = {"filename": filename, "attempts": []}

    for attempt in range(retries):
        try:
            start_time = datetime.now()
            print(f"[{start_time.strftime('%H:%M:%S')}] Попытка {attempt + 1}: {filename}")

            # Получаем метаданные
            remote_meta = get_file_metadata(filename)

            # Загружаем файл (если он уже есть, используем кеш)
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                cache_dir=cache_dir,
                force_download=False,  # Не перезаписываем, если файл уже есть
                etag_timeout=15
            )

            # Проверка целостности
            file_stat = os.stat(local_path)
            local_hash = hashlib.sha256(open(local_path, 'rb').read()).hexdigest()[:16]

            # Определяем статус
            status = "cached" if attempt == 0 else "redownloaded"
            if file_stat.st_size != remote_meta["size"]:
                status = "size_mismatch"
                raise ValueError("Размер файла не совпадает")

            end_time = datetime.now()
            result.update({
                "status": status,
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "size_mb": remote_meta["size"] / 1024 / 1024,
                "local_hash": local_hash,
                "remote_etag": remote_meta["etag"],
                "local_path": local_path
            })

            print(f"Успех: {status.upper()} | {result['size_mb']:.2f} MB | {local_path}")
            return result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            result["attempts"].append({
                "time": datetime.now().isoformat(),
                "error": error_msg
            })
            print(f"Ошибка: {error_msg}")
            time.sleep(2 ** attempt)

    result["status"] = "failed"
    return result


# СТАРТ РАБОТЫ ПРИЛОЖЕНИЯ!
# Основной цикл обработки
for idx, parquet_file in enumerate(parquet_files):
    year = str(2011 + idx)
    download_result = safe_download(parquet_file)

    # Всегда добавляем файл в file_map, даже если загрузка не удалась
    file_map[year] = {
        "path": download_result.get("local_path", ""),
        "meta": {
            "remote_etag": download_result.get("remote_etag", ""),
            "local_hash": download_result.get("local_hash", ""),
            "status": download_result.get("status", "failed")
        }
    }

# Сохранение отчёта
report = {
    "timestamp": datetime.now().isoformat(),
    "cache_dir": cache_dir,
    "files": file_map
}

with open("parquet_map.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

print(f"Отчёт сохранён в parquet_map.json. Обработано {len(file_map)} файлов.")