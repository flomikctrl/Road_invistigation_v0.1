"""Download small daily Sentinel-1 VV/VH GeoTIFF crops from CDSE Process API."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
import logging
import os
from pathlib import Path
import time as time_module
from typing import Iterable

import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "satellite_tiffs"

CDSE_CLIENT_ID = os.getenv("CDSE_CLIENT_ID", "")
CDSE_CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET", "")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
BBOX_WGS84 = [44.70, 48.50, 45.45, 48.85]
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024
REQUEST_TIMEOUT = (30, 240)

EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "VH"] }],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  return [sample.VV, sample.VH];
}
"""

LOGGER = logging.getLogger("download_crops")
DAY_DELAY_SECONDS = 1.2
RATE_LIMIT_BACKOFF_SECONDS = (3, 6, 12)


class EmptyRasterError(ValueError):
    """The TIFF is readable but has no usable VV/VH backscatter signal."""


def _has_valid_radar_data(vv: np.ndarray, vh: np.ndarray) -> bool:
    """Return true if at least one pixel has finite, non-zero values in both bands."""
    vv_values = np.asarray(vv, dtype=np.float32)
    vh_values = np.asarray(vh, dtype=np.float32)
    valid_pairs = (
        np.isfinite(vv_values)
        & np.isfinite(vh_values)
        & (vv_values != 0)
        & (vh_values != 0)
    )
    return bool(np.any(valid_pairs))


def get_access_token() -> str:
    """Получение Bearer токена через Client Credentials."""
    client_id = os.getenv("CDSE_CLIENT_ID", CDSE_CLIENT_ID)
    client_secret = os.getenv("CDSE_CLIENT_SECRET", CDSE_CLIENT_SECRET)
    if not client_id or not client_secret:
        raise RuntimeError(
            "Задайте CDSE_CLIENT_ID и CDSE_CLIENT_SECRET в переменных окружения."
        )
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        print("\n--- ОШИБКА АВТОРИЗАЦИИ CDSE ---")
        print(f"Статус: {response.status_code}")
        print(f"Ответ: {response.text}")
        print("-------------------------------\n")
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Не удалось получить access_token от Copernicus.")
    return str(token)


def _process_payload(date_str: str) -> dict[str, object]:
    scene_date = date.fromisoformat(date_str)
    start = datetime.combine(scene_date, time.min, timezone.utc)
    end = datetime.combine(scene_date, time(23, 59, 59), timezone.utc)
    return {
        "input": {
            "bounds": {
                "bbox": BBOX_WGS84,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-1-grd",
                    "dataFilter": {
                        "timeRange": {
                            "from": start.isoformat(timespec="seconds").replace("+00:00", "Z"),
                            "to": end.isoformat(timespec="seconds").replace("+00:00", "Z"),
                        },
                        "acquisitionMode": "IW",
                        "polarization": "DV",
                    },
                    "processing": {
                        "orthorectify": True,
                        "backCoeff": "GAMMA0_ELLIPSOID",
                        "demInstance": "COPERNICUS_30",
                    },
                }
            ],
        },
        "evalscript": EVALSCRIPT,
        "output": {
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "responses": [
                {"identifier": "default", "format": {"type": "image/tiff"}}
            ],
        },
    }


def _is_no_data_response(response: requests.Response) -> bool:
    if response.status_code == 404:
        return True
    if response.status_code not in {400, 422}:
        return False
    text = response.text.casefold()
    return any(
        marker in text
        for marker in ("no data", "no matching", "no scenes", "no results", "no products")
    )


def _validate_tiff(content: bytes) -> None:
    import rasterio

    try:
        with rasterio.io.MemoryFile(BytesIO(content)) as memory_file:
            with memory_file.open() as dataset:
                if dataset.count != 2:
                    raise ValueError(
                        f"Ожидалось 2 канала (VV/VH), получено {dataset.count}."
                    )
                if dataset.width != OUTPUT_WIDTH or dataset.height != OUTPUT_HEIGHT:
                    raise ValueError(
                        f"Неожиданный размер TIFF: {dataset.width}x{dataset.height}."
                    )
                vv = dataset.read(1, masked=True).filled(np.nan)
                vh = dataset.read(2, masked=True).filled(np.nan)
                if not _has_valid_radar_data(vv, vh):
                    raise EmptyRasterError(
                        "В GeoTIFF нет парных валидных VV/VH пикселей."
                    )
                for band_index in (1, 2):
                    band = dataset.read(band_index, masked=True)
                    values = np.asarray(band.compressed(), dtype=np.float32)
                    finite_values = values[np.isfinite(values)]
                    if finite_values.size == 0 or not np.any(finite_values != 0):
                        raise EmptyRasterError(
                            f"Канал {band_index} TIFF пустой (только нули/NaN)."
                        )
    except EmptyRasterError:
        raise
    except rasterio.errors.RasterioError as error:
        raise ValueError("Ответ Copernicus не является валидным GeoTIFF.") from error


def download_scene_for_date(date_str: str, token: str) -> Path | None:
    """Скачивание одного кропа Sentinel-1 или пропуск, если пролета не было."""
    date.fromisoformat(date_str)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"s1_{date_str}.tif"
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            _validate_tiff(output_path.read_bytes())
        except (EmptyRasterError, ValueError):
            LOGGER.warning("Удаляю пустой/поврежденный TIFF %s", output_path.name)
            output_path.unlink(missing_ok=True)
        else:
            LOGGER.info("Файл уже существует и содержит сигнал: %s", output_path.name)
            return output_path

    request_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff",
    }
    payload = _process_payload(date_str)
    for retry_index in range(len(RATE_LIMIT_BACKOFF_SECONDS) + 1):
        response = requests.post(
            PROCESS_URL,
            json=payload,
            headers=request_headers,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 429:
            break
        if retry_index == len(RATE_LIMIT_BACKOFF_SECONDS):
            response.raise_for_status()
        delay = RATE_LIMIT_BACKOFF_SECONDS[retry_index]
        LOGGER.warning(
            "CDSE rate limit for %s (429); retry %d/%d in %d seconds.",
            date_str,
            retry_index + 1,
            len(RATE_LIMIT_BACKOFF_SECONDS),
            delay,
        )
        time_module.sleep(delay)

    if _is_no_data_response(response):
        LOGGER.info("Пролета над поймой за %s не было (пропуск).", date_str)
        return None
    response.raise_for_status()
    try:
        _validate_tiff(response.content)
    except EmptyRasterError as error:
        LOGGER.warning("CDSE returned an empty crop for %s: %s", date_str, error)
        return None

    temporary_path = output_path.with_suffix(".tif.part")
    temporary_path.write_bytes(response.content)
    temporary_path.replace(output_path)
    LOGGER.info("✅ Скачан снимок: %s (%0.1f МБ)", output_path.name, output_path.stat().st_size / 1048576)
    return output_path


def _date_range(start_value: str | date, end_value: str | date) -> Iterable[str]:
    start_date = date.fromisoformat(start_value) if isinstance(start_value, str) else start_value
    end_date = date.fromisoformat(end_value) if isinstance(end_value, str) else end_value
    if end_date < start_date:
        raise ValueError("Дата окончания раньше даты старта.")
    current = start_date
    while current <= end_date:
        yield current.isoformat()
        current += timedelta(days=1)


def batch_download(date_ranges: Iterable[tuple[str | date, str | date]]) -> list[Path]:
    """Пакетное скачивание снимков за указанные периоды."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Авторизация в Copernicus Data Space через Client Credentials...")
    token = get_access_token()
    LOGGER.info("Токен успешно получен. Начинаем загрузку снимков...")

    downloaded: list[Path] = []
    for start_date, end_date in date_ranges:
        for date_str in _date_range(start_date, end_date):
            try:
                path = download_scene_for_date(date_str, token)
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code in {401, 403}:
                    LOGGER.info("Токен истек, обновляем...")
                    token = get_access_token()
                    path = download_scene_for_date(date_str, token)
                else:
                    LOGGER.exception("Сбой запроса за %s", date_str)
                    continue
            except Exception:
                LOGGER.exception("Ошибка при обработке даты %s", date_str)
                continue
            finally:
                time_module.sleep(DAY_DELAY_SECONDS)
            if path is not None:
                downloaded.append(path)
    LOGGER.info("Загрузка завершена! Всего скачано снимков: %d.", len(downloaded))
    return downloaded


def main() -> None:
    batch_download(
        [
            ("2018-01-01", "2026-10-02"),
        ]
    )


if __name__ == "__main__":
    main()