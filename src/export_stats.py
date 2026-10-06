"""Export two-column passability statistics for processed Sentinel-1 dates."""

from __future__ import annotations

from contextlib import closing
from datetime import date
from io import StringIO
from pathlib import Path
import sqlite3
from collections import Counter
from typing import Any

import pandas as pd

from src.router import DEFAULT_DB_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "flood_road_status_2018_2026.csv"
DB_PATH = DEFAULT_DB_PATH
CSV_COLUMNS = ("Дата", "Проезд через пойму")
PERMANENT_WATER_FREQUENCY = 0.5
DYNAMIC_CLOSED_THRESHOLD = 40
SUMMARY_CLOSED_THRESHOLD = 380


def _tables(connection: sqlite3.Connection) -> dict[str, set[str]]:
    table_names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    ]
    result = {}
    for table_name in table_names:
        columns = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        result[table_name] = {str(column[1]).casefold() for column in columns}
    return result


def _normalize_date(value: Any) -> str | None:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError):
        return None


def _processed_dates(
    connection: sqlite3.Connection, tables: dict[str, set[str]]
) -> list[str]:
    if "flood_polygons" in tables and "date" in tables["flood_polygons"]:
        raw_dates = [row[0] for row in connection.execute("SELECT date FROM flood_polygons")]
    else:
        date_table = next(
            (
                table
                for table, columns in tables.items()
                if "date" in columns and table.casefold() in {"flooded_edges", "daily_flooded_edges"}
            ),
            None,
        )
        if date_table is None:
            return []
        raw_dates = [
            row[0]
            for row in connection.execute(f'SELECT DISTINCT date FROM "{date_table}"')
        ]
    return sorted({normalized for value in raw_dates if (normalized := _normalize_date(value))})


def _edge_history(
    connection: sqlite3.Connection, tables: dict[str, set[str]]
) -> list[tuple[str, int]] | None:
    for table_name in ("flooded_edges", "daily_flooded_edges"):
        columns = tables.get(table_name)
        if columns and {"date", "edge_id"} <= columns:
            return [
                (date_text, int(edge_id))
                for raw_date, edge_id in connection.execute(
                    f'SELECT date, edge_id FROM "{table_name}"'
                )
                if (date_text := _normalize_date(raw_date)) is not None
                and edge_id is not None
            ]
    return None


def _summary_history(
    connection: sqlite3.Connection, tables: dict[str, set[str]]
) -> dict[str, int] | None:
    summary_columns = {
        "flooded_count",
        "flood_count",
        "flooded_edge_count",
        "flooded_edges_count",
        "blocked_count",
        "blocked_edge_count",
        "edge_count",
        "count",
        "count_edges",
        "flooded_edges",
    }
    for table_name, columns in tables.items():
        if "date" not in columns or "edge_id" in columns:
            continue
        count_column = next(
            (column for column in summary_columns if column in columns), None
        )
        if count_column is None:
            continue
        result = {}
        for raw_date, count in connection.execute(
            f'SELECT date, "{count_column}" FROM "{table_name}"'
        ):
            date_text = _normalize_date(raw_date)
            if date_text is not None:
                result[date_text] = int(count or 0)
        return result
    return None


def build_status_dataframe(
    db_path: str | Path = DB_PATH,
) -> pd.DataFrame:
    """Apply permanent-water baseline and return exactly two status columns."""
    db_path = Path(db_path)
    if not db_path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)

    with closing(sqlite3.connect(db_path)) as connection:
        tables = _tables(connection)
        dates = _processed_dates(connection, tables)
        edge_history = _edge_history(connection, tables)
        summary_history = None if edge_history is not None else _summary_history(connection, tables)

    if edge_history is not None:
        observations = len(dates) or len({date_text for date_text, _edge_id in edge_history})
        frequency: Counter[int] = Counter()
        edges_by_date: dict[str, set[int]] = {}
        for date_text, edge_id in edge_history:
            edge_set = edges_by_date.setdefault(date_text, set())
            if edge_id not in edge_set:
                edge_set.add(edge_id)
                frequency[edge_id] += 1
        permanent_edges = {
            edge_id
            for edge_id, count in frequency.items()
            if observations and count / observations > PERMANENT_WATER_FREQUENCY
        }
        dynamic_counts = {
            date_text: sum(
                edge_id not in permanent_edges for edge_id in edges_by_date.get(date_text, set())
            )
            for date_text in dates
        }
        rows = [
            {
                "Дата": date_text,
                "Проезд через пойму": (
                    "Проезда нет"
                    if dynamic_counts.get(date_text, 0) > DYNAMIC_CLOSED_THRESHOLD
                    else "Есть проезд"
                ),
            }
            for date_text in dates
        ]
    elif summary_history is not None:
        dates = sorted(set(dates) | set(summary_history))
        rows = [
            {
                "Дата": date_text,
                "Проезд через пойму": (
                    "Проезда нет"
                    if summary_history.get(date_text, 0) > SUMMARY_CLOSED_THRESHOLD
                    else "Есть проезд"
                ),
            }
            for date_text in dates
        ]
    else:
        rows = [{"Дата": date_text, "Проезд через пойму": "Есть проезд"} for date_text in dates]
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def export_status_csv(
    dataframe: pd.DataFrame,
    output_path: str | Path = CSV_PATH,
) -> bytes:
    """Write UTF-8-SIG CSV to disk and return its exact bytes for Streamlit download."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream = StringIO(newline="")
    dataframe.loc[:, list(CSV_COLUMNS)].to_csv(stream, index=False, lineterminator="\n")
    content = stream.getvalue().encode("utf-8-sig")
    output_path.write_bytes(content)
    return content


def main() -> None:
    dataframe = build_status_dataframe()
    export_status_csv(dataframe)
    print(f"Exported {len(dataframe)} processed dates to {CSV_PATH}")


if __name__ == "__main__":
    main()
