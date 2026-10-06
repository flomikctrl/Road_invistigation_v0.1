"""LLM function schema and dispatcher for flood-aware route checks."""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import osmnx as ox

from src.router import DEFAULT_DB_PATH, check_route_instant


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "roads_graph.graphml"

ROUTE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_flood_route",
        "description": (
            "Проверить, проходим ли маршрут между двумя населенными пунктами "
            "по незатопленным дорогам на указанную дату."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Название начального населенного пункта",
                },
                "destination": {
                    "type": "string",
                    "description": "Название конечного населенного пункта",
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Дата поездки в формате YYYY-MM-DD",
                },
            },
            "required": ["origin", "destination", "date"],
            "additionalProperties": False,
        },
    },
}

SETTLEMENT_COORDS: dict[str, tuple[float, float]] = {
    "волжский": (48.7858, 44.7797),
    "средняя ахтуба": (48.7110, 44.8670),
    "ленинск": (48.6930, 45.1990),
    "покровка": (48.6200, 45.0800),
}


@lru_cache(maxsize=1)
def load_graph(graph_path: str = str(DEFAULT_GRAPH_PATH)) -> Any:
    """Load and cache the prebuilt road graph for the process lifetime."""
    return ox.load_graphml(filepath=graph_path)


def _resolve_settlement(name: str) -> tuple[float, float]:
    normalized_name = " ".join(name.casefold().split())
    try:
        return SETTLEMENT_COORDS[normalized_name]
    except KeyError as error:
        choices = ", ".join(sorted(SETTLEMENT_COORDS))
        raise ValueError(f"Неизвестный населенный пункт. Доступны: {choices}") from error


def handle_route_tool(
    arguments: Mapping[str, Any],
    graph: Any | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Resolve tool arguments, then run the in-memory graph route check."""
    origin_name = arguments.get("origin")
    destination_name = arguments.get("destination")
    date_str = arguments.get("date")
    if not all(isinstance(value, str) for value in (origin_name, destination_name, date_str)):
        raise ValueError("Ожидаются строковые origin, destination и date.")

    date.fromisoformat(date_str)
    origin_coords = _resolve_settlement(origin_name)
    destination_coords = _resolve_settlement(destination_name)
    road_graph = graph if graph is not None else load_graph()
    result = check_route_instant(
        origin_coords,
        destination_coords,
        date_str,
        road_graph,
        db_path,
    )
    return {
        **result,
        "origin": origin_name,
        "destination": destination_name,
        "date": date_str,
    }


def dispatch_tool(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch a model function call by its schema name."""
    if name != "check_flood_route":
        raise ValueError(f"Неизвестный инструмент: {name}")
    return handle_route_tool(arguments)