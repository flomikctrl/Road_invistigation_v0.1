"""Interactive flood-aware road accessibility map."""

from contextlib import closing
from datetime import date
from pathlib import Path
import json
import sqlite3
from typing import Any

import folium
import networkx as nx
import osmnx as ox
import streamlit as st
from shapely.geometry import LineString, mapping
from shapely import wkt
from streamlit_folium import st_folium

from agent_tool import SETTLEMENT_COORDS, handle_route_tool
from src.export_stats import CSV_PATH, build_status_dataframe, export_status_csv
from src.graph_builder import build_road_graph
from src.router import DEFAULT_DB_PATH


PROJECT_ROOT = Path(__file__).resolve().parent
GRAPH_PATH = PROJECT_ROOT / "data" / "roads_graph.graphml"
DB_PATH = DEFAULT_DB_PATH
MAP_CENTER = (48.71, 44.95)

st.set_page_config(
    page_title="Доступность дорог | Волго-Ахтубинская пойма",
    page_icon="🗺️",
    layout="wide",
)

@st.cache_resource(show_spinner="Загружаю дорожный граф…")
def load_road_graph() -> Any:
    if not GRAPH_PATH.exists():
        return build_road_graph(GRAPH_PATH)
    return ox.load_graphml(filepath=GRAPH_PATH)


def _edge_id(data: dict[str, Any]) -> int | None:
    value = data.get("edge_id")
    return int(value) if value is not None else None


def _edge_geometry(graph: Any, u: Any, v: Any, data: dict[str, Any]) -> LineString:
    geometry = data.get("geometry")
    if isinstance(geometry, str):
        geometry = wkt.loads(geometry)
    if geometry is not None and hasattr(geometry, "coords"):
        return geometry
    return LineString(
        [
            (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
            (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
        ]
    )


def _feature(edge_id: int | None, geometry: LineString, **properties: Any) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": {"edge_id": edge_id, **properties},
    }


def _all_road_features(graph: Any) -> list[dict[str, Any]]:
    features = []
    for u, v, _key, data in graph.edges(keys=True, data=True):
        features.append(_feature(_edge_id(data), _edge_geometry(graph, u, v, data)))
    return features


def _get_flooded_ids(db_path: Path, date_str: str) -> set[int]:
    with closing(sqlite3.connect(db_path)) as connection:
        rows = connection.execute(
            "SELECT edge_id FROM daily_flooded_edges WHERE date = ?", (date_str,)
        )
        return {int(row[0]) for row in rows}


def _route_path(
    graph: Any,
    origin: tuple[float, float],
    destination: tuple[float, float],
    blocked_ids: set[int],
) -> tuple[str, list[Any]]:
    origin_node = ox.distance.nearest_nodes(graph, X=origin[1], Y=origin[0])
    destination_node = ox.distance.nearest_nodes(graph, X=destination[1], Y=destination[0])

    def edge_is_open(u: Any, v: Any, key: Any = None) -> bool:
        data = graph.edges[u, v, key] if graph.is_multigraph() else graph.edges[u, v]
        return _edge_id(data) not in blocked_ids

    baseline = nx.shortest_path(graph, origin_node, destination_node, weight="length")
    available = nx.subgraph_view(graph, filter_edge=edge_is_open)
    path = nx.shortest_path(available, origin_node, destination_node, weight="length")
    return ("PASSABLE" if path == baseline else "DETOUR", path)


def _path_features(graph: Any, path: list[Any], blocked_ids: set[int]) -> list[dict[str, Any]]:
    features = []
    for u, v in zip(path, path[1:]):
        if graph.is_multigraph():
            choices = [
                (key, data)
                for key, data in graph[u][v].items()
                if _edge_id(data) not in blocked_ids
            ]
            _key, data = min(choices, key=lambda item: float(item[1].get("length", 1.0)))
        else:
            data = graph.edges[u, v]
        features.append(_feature(_edge_id(data), _edge_geometry(graph, u, v, data)))
    return features


def build_map(
    graph: Any,
    date_str: str,
    blocked_ids: set[int],
    corridors: dict[str, dict[str, Any]],
    water_geojson: dict[str, Any] | None = None,
    route_features: list[dict[str, Any]] | None = None,
    origin: tuple[float, float] | None = None,
    destination: tuple[float, float] | None = None,
) -> folium.Map:
    map_object = folium.Map(
        location=MAP_CENTER,
        zoom_start=10,
        tiles=None,
        control_scale=True,
        attribution_control=False,
        attributionControl=False,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Спутник · Esri World Imagery",
        overlay=False,
        control=True,
    ).add_to(map_object)
    folium.TileLayer(
        "OpenStreetMap", name="Схема · OpenStreetMap", overlay=False, control=True
    ).add_to(map_object)

    if water_geojson and water_geojson.get("geometry"):
        folium.GeoJson(
            water_geojson,
            name=f"Вода по Sentinel-1 · {date_str}",
            style_function=lambda _feature: {
                "color": "#168bd2",
                "weight": 1,
                "fillColor": "#168bd2",
                "fillOpacity": 0.4,
                "opacity": 0.7,
            },
        ).add_to(map_object)

    roads = _all_road_features(graph)
    if roads:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": roads},
            name="Дорожная сеть OSM",
            style_function=lambda _feature: {
                "color": "#f5f3e9",
                "weight": 1.3,
                "opacity": 0.58,
            },
            show=True,
        ).add_to(map_object)

    flooded_features = [
        _feature(_edge_id(data), _edge_geometry(graph, u, v, data))
        for u, v, _key, data in graph.edges(keys=True, data=True)
        if _edge_id(data) in blocked_ids
    ]
    if flooded_features:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": flooded_features},
            name=f"Затопленные коридоры · {date_str}",
            style_function=lambda _feature: {
                "color": "#e53935",
                "weight": 5,
                "opacity": 0.95,
            },
        ).add_to(map_object)

    if route_features:
        route_color = "#f08a24" if route_features[0].get("properties", {}).get("status") == "DETOUR" else "#22a06b"
        folium.GeoJson(
            {"type": "FeatureCollection", "features": route_features},
            name="Маршрут",
            style_function=lambda _feature: {
                "color": route_color,
                "weight": 6,
                "opacity": 0.95,
            },
        ).add_to(map_object)

    if origin:
        folium.Marker(origin, tooltip="Точка А", icon=folium.Icon(color="green", icon="play")).add_to(map_object)
    if destination:
        folium.Marker(destination, tooltip="Точка Б", icon=folium.Icon(color="red", icon="flag")).add_to(map_object)

    folium.LayerControl(collapsed=False).add_to(map_object)
    return map_object


WAYPOINT_COORDS = {
    "Заплавное": (48.735, 45.080),
    "Суходол": (48.700, 45.030),
    "Чапаевец": (48.655, 45.150),
}

CORRIDOR_SPECS = (
    ("volzhsky_srednyaya", "Волжский ↔ Средняя Ахтуба · 18Р-1 / пр. Ленина", "волжский", "средняя ахтуба", ()),
    ("srednyaya_leninsk", "Средняя Ахтуба ↔ Ленинск · 18Р-1 через Заплавное", "средняя ахтуба", "ленинск", ("Заплавное",)),
    ("leninsk_pokrovka", "Ленинск ↔ Покровка · восточный объезд", "ленинск", "покровка", ()),
    ("srednyaya_pokrovka_floodplain", "Пойменный путь · Средняя Ахтуба ↔ Покровка через Суходол и Чапаевец", "средняя ахтуба", "покровка", ("Суходол", "Чапаевец")),
    ("volzhsky_pokrovka_direct", "Прямой ход · Волжский ↔ Покровка", "волжский", "покровка", ()),
)


def _osmids(data: dict[str, Any]) -> tuple[str, ...]:
    value = data.get("osmid")
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    return tuple(sorted(str(item) for item in values))


def build_key_corridors(graph: Any) -> dict[str, dict[str, Any]]:
    corridors = {}
    path_cache = {}
    for key, name, origin_name, destination_name, waypoint_names in CORRIDOR_SPECS:
        points = [
            SETTLEMENT_COORDS[origin_name],
            *(WAYPOINT_COORDS[item] for item in waypoint_names),
            SETTLEMENT_COORDS[destination_name],
        ]
        nodes = []
        for latitude, longitude in points:
            nodes.append(ox.distance.nearest_nodes(graph, X=longitude, Y=latitude))
        path = []
        for start_node, end_node in zip(nodes, nodes[1:]):
            pair = (start_node, end_node)
            if pair not in path_cache:
                path_cache[pair] = nx.shortest_path(
                    graph, start_node, end_node, weight="length"
                )
            leg = path_cache[pair]
            path.extend(leg if not path else leg[1:])

        edge_ids: set[int] = set()
        geometries = []
        length_m = 0.0
        for u, v in zip(path, path[1:]):
            if graph.is_multigraph():
                data = min(
                    graph[u][v].values(),
                    key=lambda attributes: float(attributes.get("length", 1.0)),
                )
            else:
                data = graph.edges[u, v]
            edge_id = _edge_id(data)
            if edge_id is not None:
                edge_ids.add(edge_id)
            osmids = _osmids(data)
            for start, end in ((u, v), (v, u)):
                for reverse_data in graph.get_edge_data(start, end, default={}).values():
                    if osmids and _osmids(reverse_data) == osmids:
                        reverse_id = _edge_id(reverse_data)
                        if reverse_id is not None:
                            edge_ids.add(reverse_id)
            length_m += float(data.get("length", 0.0))
            geometries.append(_edge_geometry(graph, u, v, data))
        corridors[key] = {
            "key": key,
            "name": name,
            "nodes": path,
            "edge_ids": edge_ids,
            "geometries": geometries,
            "length_km": length_m / 1000,
        }
    return corridors


@st.cache_resource(show_spinner="Собираю пять коридоров OSM…")
def load_corridors() -> dict[str, dict[str, Any]]:
    return build_key_corridors(load_road_graph())


def _processed_dates(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    with closing(sqlite3.connect(db_path)) as connection:
        try:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT date FROM flood_polygons ORDER BY date DESC"
                )
            ]
        except sqlite3.OperationalError:
            return []


def _observation_for_date(
    db_path: Path, date_str: str
) -> tuple[dict[str, Any] | None, set[int]]:
    with closing(sqlite3.connect(db_path)) as connection:
        try:
            row = connection.execute(
                "SELECT geojson FROM flood_polygons WHERE date = ?", (date_str,)
            ).fetchone()
            edge_rows = connection.execute(
                "SELECT edge_id FROM daily_flooded_edges WHERE date = ?", (date_str,)
            ).fetchall()
        except sqlite3.OperationalError:
            return None, set()
    water_geojson = json.loads(row[0]) if row and row[0] else None
    if water_geojson and water_geojson.get("type") == "FeatureCollection":
        water_geojson = next(
            (feature for feature in water_geojson.get("features", []) if feature.get("geometry")),
            None,
        )
    if water_geojson and water_geojson.get("type") == "Feature":
        water_geojson = {
            "type": "Feature",
            "geometry": water_geojson.get("geometry"),
            "properties": water_geojson.get("properties", {}),
        }
    return water_geojson, {int(edge_row[0]) for edge_row in edge_rows}


def _archive_rows(
    db_path: Path,
    corridors: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    dates = _processed_dates(db_path)
    rows = []
    for date_str in sorted(dates):
        _water, flooded_ids = _observation_for_date(db_path, date_str)
        flooded_corridors = [
            corridor["name"]
            for corridor in corridors.values()
            if corridor["edge_ids"] & flooded_ids
        ]
        floodplain = corridors["srednyaya_pokrovka_floodplain"]
        rows.append(
            {
                "Дата съемки": date_str,
                "Затопленных ребер дорог": len(flooded_ids),
                "Проезд через пойму": "Затоплен" if floodplain["edge_ids"] & flooded_ids else "Открыт",
                "Затопленные коридоры": "; ".join(flooded_corridors) if flooded_corridors else "Нет",
            }
        )
    return rows


def _key_edge_rows(
    graph: Any,
    corridors: dict[str, dict[str, Any]],
    flooded_ids: set[int],
) -> list[dict[str, Any]]:
    rows = []
    for corridor in corridors.values():
        is_flooded = bool(corridor["edge_ids"] & flooded_ids)
        alternative = "Объезд не требуется"
        if is_flooded:
            if corridor["key"] == "srednyaya_pokrovka_floodplain":
                detour_ids = corridors["srednyaya_leninsk"]["edge_ids"] | corridors["leninsk_pokrovka"]["edge_ids"]
                alternative = (
                    "Через Ленинск"
                    if not detour_ids & flooded_ids
                    else _alternative_route_label(graph, corridor, flooded_ids)
                )
            else:
                alternative = _alternative_route_label(graph, corridor, flooded_ids)
        rows.append(
            {
                "Маршрут": corridor["name"],
                "Длина, км": round(corridor["length_km"], 2),
                "Состояние": "🔴 Затоплен" if is_flooded else "🟢 Открыт",
                "Альтернатива": alternative,
            }
        )
    return rows


def _alternative_route_label(
    graph: Any,
    corridor: dict[str, Any],
    flooded_ids: set[int],
) -> str:
    start_node = corridor["nodes"][0]
    end_node = corridor["nodes"][-1]

    def edge_is_open(u: Any, v: Any, key: Any = None) -> bool:
        data = graph.edges[u, v, key] if graph.is_multigraph() else graph.edges[u, v]
        return _edge_id(data) not in flooded_ids

    available_graph = nx.subgraph_view(graph, filter_edge=edge_is_open)
    try:
        path = nx.shortest_path(
            available_graph, start_node, end_node, weight="length"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return "Сухого объезда нет"

    leninsk_coords = SETTLEMENT_COORDS["ленинск"]
    leninsk_node = ox.distance.nearest_nodes(
        graph, X=leninsk_coords[1], Y=leninsk_coords[0]
    )
    if leninsk_node in path and leninsk_node not in (start_node, end_node):
        return "Объезд через Ленинск доступен"
    return "Объезд по сухим дорогам OSM доступен"


st.title("Доступность дорог Волго-Ахтубинской поймы")
st.caption("Результаты обработки реальных Sentinel-1 снимков и OSM-дорожной сети")

processed_dates = _processed_dates(DB_PATH)
if not processed_dates:
    st.info("В базе нет обработанных снимков. Запустите download_crops.py и run_inference.py")
    st.stop()

try:
    with st.spinner("Загружаю граф дорог и коридоры…"):
        road_graph = load_road_graph()
        corridors = load_corridors()
except Exception as error:
    st.error(f"Не удалось загрузить дорожные данные: {error}")
    st.stop()

date_str = st.selectbox(
    "Дата обработанного снимка",
    processed_dates,
    format_func=lambda value: date.fromisoformat(value).strftime("%d.%m.%Y"),
)
water_geojson, flooded_ids = _observation_for_date(DB_PATH, date_str)
map_tab, archive_tab, matrix_tab = st.tabs(
    [
        "Интерактивная карта и маршрутизация",
        "Архив спутниковых съемок",
        "Матрица ключевых коридоров",
    ]
)

with map_tab:
    st.subheader("Маршрут между населенными пунктами")
    origin_col, destination_col, route_col = st.columns([1, 1, 1.25])
    settlement_names = list(SETTLEMENT_COORDS)
    with origin_col:
        origin_name = st.selectbox(
            "Точка А", settlement_names, index=0, format_func=str.title
        )
    with destination_col:
        destination_name = st.selectbox(
            "Точка Б", settlement_names, index=3, format_func=str.title
        )
    with route_col:
        st.markdown(" ")
        check_route = st.button(
            "Проверить доступность", type="primary", width="stretch"
        )

    route_key = "real_data_route"
    if check_route:
        try:
            route_result = handle_route_tool(
                {"origin": origin_name, "destination": destination_name, "date": date_str},
                graph=road_graph,
                db_path=DB_PATH,
            )
            if route_result["status"] != "BLOCKED":
                _path_status, route_nodes = _route_path(
                    road_graph,
                    SETTLEMENT_COORDS[origin_name],
                    SETTLEMENT_COORDS[destination_name],
                    flooded_ids,
                )
                route_result["route_nodes"] = route_nodes
                route_result["route_features"] = _path_features(
                    road_graph, route_nodes, flooded_ids
                )
                for feature in route_result["route_features"]:
                    feature["properties"]["status"] = route_result["status"]
            st.session_state[route_key] = {
                "date": date_str,
                "origin": origin_name,
                "destination": destination_name,
                "result": route_result,
            }
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            st.session_state[route_key] = {
                "date": date_str,
                "origin": origin_name,
                "destination": destination_name,
                "result": {"status": "BLOCKED", "distance_km": None},
            }
        except Exception as error:
            st.error(f"Ошибка проверки маршрута: {error}")

    route_state = st.session_state.get(route_key)
    if route_state and (
        route_state["date"], route_state["origin"], route_state["destination"]
    ) == (date_str, origin_name, destination_name):
        route_result = route_state["result"]
        if route_result["status"] == "BLOCKED":
            st.error("Маршрут закрыт: пути по незатопленным дорогам нет.")
        else:
            route_label = "Прямой путь" if route_result["status"] == "PASSABLE" else "Объезд"
            if route_result["status"] == "DETOUR":
                leninsk_coords = SETTLEMENT_COORDS["ленинск"]
                leninsk_node = ox.distance.nearest_nodes(
                    road_graph, X=leninsk_coords[1], Y=leninsk_coords[0]
                )
                if leninsk_node in route_result.get("route_nodes", []):
                    route_label = "Объезд через Ленинск"
            st.success(
                f"Проезд открыт · {route_label} · {route_result['distance_km']:.2f} км"
            )
        if corridors["srednyaya_pokrovka_floodplain"]["edge_ids"] & flooded_ids:
            st.warning("Пойменный коридор через Суходол/Чапаевец перекрыт водой.")
    else:
        route_result = None

    route_features = (
        route_result.get("route_features")
        if route_result and route_result["status"] != "BLOCKED"
        else None
    )
    map_object = build_map(
        road_graph,
        date_str,
        flooded_ids,
        corridors,
        water_geojson=water_geojson,
        route_features=route_features,
        origin=SETTLEMENT_COORDS[origin_name] if route_features else None,
        destination=SETTLEMENT_COORDS[destination_name] if route_features else None,
    )
    st_folium(map_object, use_container_width=True, height=690, returned_objects=[])
    st.caption(
        "Красные линии: затопленная часть ребра не менее 25% длины или более 50 м; "
        "мосты OSM исключаются. Дороги: © OpenStreetMap contributors · Подложка: © Esri"
    )

with archive_tab:
    st.subheader("Обработанные спутниковые съемки")
    archive_rows = _archive_rows(DB_PATH, corridors)
    st.dataframe(archive_rows, width="stretch", hide_index=True)
    st.caption("В архив включены только даты, присутствующие в таблице flood_polygons.")
    status_dataframe = build_status_dataframe(DB_PATH)
    csv_content = export_status_csv(status_dataframe, CSV_PATH)
    st.download_button(
        "Скачать CSV: дата и проезд через пойму",
        data=csv_content,
        file_name=CSV_PATH.name,
        mime="text/csv",
        width="content",
    )

with matrix_tab:
    st.subheader(f"Статус пяти коридоров · {date_str}")
    matrix_rows = _key_edge_rows(road_graph, corridors, flooded_ids)
    st.dataframe(matrix_rows, width="stretch", hide_index=True)