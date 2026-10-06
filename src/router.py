"""Fast connectivity checks on a preloaded OSM road graph."""

from pathlib import Path
import sqlite3
from contextlib import closing
from typing import Any, Literal, TypedDict

import networkx as nx
import osmnx as ox


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "floods.db"
RouteStatus = Literal["PASSABLE", "DETOUR", "BLOCKED"]


class RouteResult(TypedDict):
    status: RouteStatus
    distance_km: float | None


def _read_flooded_edge_ids(date_str: str, db_path: str | Path) -> set[int]:
    with closing(
        sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    ) as connection:
        rows = connection.execute(
            "SELECT edge_id FROM daily_flooded_edges WHERE date = ?",
            (date_str,),
        )
        return {int(row[0]) for row in rows}


def _edge_id(data: dict[str, Any]) -> int | None:
    value = data.get("edge_id")
    return int(value) if value is not None else None


def _path_length_m(graph: Any, path: list[Any], flooded_edge_ids: set[int]) -> float:
    length_m = 0.0
    for u, v in zip(path, path[1:]):
        if graph.is_multigraph():
            parallel_edges = graph[u][v].values()
            available_lengths = [
                float(data.get("length", 1.0))
                for data in parallel_edges
                if _edge_id(data) not in flooded_edge_ids
            ]
            if not available_lengths:
                raise nx.NetworkXNoPath(f"No unflooded edge connects {u} to {v}")
            length_m += min(available_lengths)
        else:
            data = graph.edges[u, v]
            if _edge_id(data) in flooded_edge_ids:
                raise nx.NetworkXNoPath(f"Road edge {u}-{v} is flooded")
            length_m += float(data.get("length", 1.0))
    return length_m


def check_route_instant(
    origin_coords: tuple[float, float],
    dest_coords: tuple[float, float],
    date_str: str,
    graph: Any,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> RouteResult:
    """Check connectivity for (latitude, longitude) coordinates and an ISO date."""
    origin_node = ox.distance.nearest_nodes(
        graph, X=float(origin_coords[1]), Y=float(origin_coords[0])
    )
    destination_node = ox.distance.nearest_nodes(
        graph, X=float(dest_coords[1]), Y=float(dest_coords[0])
    )
    flooded_edge_ids = _read_flooded_edge_ids(date_str, db_path)

    try:
        baseline_path = nx.shortest_path(graph, origin_node, destination_node, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {"status": "BLOCKED", "distance_km": None}

    def edge_is_open(u: Any, v: Any, key: Any = None) -> bool:
        if graph.is_multigraph():
            data = graph.edges[u, v, key]
        else:
            data = graph.edges[u, v]
        return _edge_id(data) not in flooded_edge_ids

    available_graph = nx.subgraph_view(graph, filter_edge=edge_is_open)
    try:
        available_path = nx.shortest_path(
            available_graph, origin_node, destination_node, weight="length"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {"status": "BLOCKED", "distance_km": None}

    status: RouteStatus = "PASSABLE" if available_path == baseline_path else "DETOUR"
    distance_km = _path_length_m(graph, available_path, flooded_edge_ids) / 1000
    return {"status": status, "distance_km": round(distance_km, 3)}