"""Build and persist the OSM road graph for the test region."""

from pathlib import Path
from typing import Any

import osmnx as ox


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "roads_graph.graphml"

# Input bounds are south, west, north, east; OSMnx 2 expects west, south, east, north.
REGION_BOUNDS = (48.50, 44.70, 48.85, 45.45)


def build_road_graph(output_path: str | Path = DEFAULT_GRAPH_PATH) -> Any:
    """Download the region's drivable network, assign stable edge IDs, and save it."""
    south, west, north, east = REGION_BOUNDS
    graph = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        network_type="drive",
        simplify=True,
    )

    edges = sorted(
        graph.edges(keys=True),
        key=lambda edge: (str(edge[0]), str(edge[1]), str(edge[2])),
    )
    for edge_id, (u, v, key) in enumerate(edges):
        graph.edges[u, v, key]["edge_id"] = edge_id

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=destination)
    return graph


if __name__ == "__main__":
    built_graph = build_road_graph()
    print(f"Saved {built_graph.number_of_edges()} road edges to {DEFAULT_GRAPH_PATH}")