"""Segment water in Sentinel-1 VV/VH crops and index intersecting road edges."""

from __future__ import annotations

from contextlib import closing
from datetime import date
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from rasterio.features import shapes
from scipy.ndimage import binary_closing, binary_opening, median_filter
from shapely.geometry import GeometryCollection, shape
from shapely.ops import transform, unary_union
from pyproj import Transformer

from src.graph_builder import DEFAULT_GRAPH_PATH, build_road_graph
from src.router import DEFAULT_DB_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIFF_DIR = PROJECT_ROOT / "data" / "satellite_tiffs"
MASK_DIR = PROJECT_ROOT / "data" / "water_masks"
WEIGHTS_PATH = PROJECT_ROOT / "models" / "unet_water.pth"
DB_PATH = DEFAULT_DB_PATH
WATER_THRESHOLD_DB = -15.5
MORPHOLOGY_SIZE = 3
VV_WATER_MAX_DB = -16.0
VH_WATER_MAX_DB = -23.5
VV_VH_MIN_DIFFERENCE_DB = 3.0
MIN_WATER_POLYGON_AREA_M2 = 500.0
MIN_EDGE_FLOOD_FRACTION = 0.25
MIN_EDGE_FLOOD_LENGTH_M = 50.0
METRIC_LENGTH_TOLERANCE_M = 0.01
METRIC_CRS = "EPSG:32638"
LOGGER = logging.getLogger("run_inference")
DATE_PATTERN = re.compile(r"^s1_(\d{4}-\d{2}-\d{2})$")


def load_model(
    weights_path: str | Path = WEIGHTS_PATH,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Load an explicitly requested trained water model; never return random weights."""
    import segmentation_models_pytorch as smp
    import torch

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Trained water checkpoint is missing: {weights_path}. "
            "Default inference uses the Sentinel-1 SAR threshold detector."
        )
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
        activation="sigmoid",
    )
    try:
        checkpoint = torch.load(weights_path, map_location=selected_device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=selected_device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported model checkpoint format: {weights_path}")
    checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.to(selected_device)
    model.eval()
    LOGGER.info("Loaded trained water segmentation checkpoint: %s", weights_path)
    return model, selected_device


def _to_db(channel: np.ndarray) -> np.ndarray:
    power = np.clip(np.asarray(channel, dtype=np.float32), 1e-6, 1.0)
    return (10.0 * np.log10(power)).astype(np.float32)


def normalize_vv_vh(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """Convert positive linear VV/VH backscatter to dB for optional trained U-Net use."""
    vv_db = _to_db(vv)
    vh_db = _to_db(vh)
    normalized_vv = np.clip((vv_db + 15.0) / 8.0, -5.0, 5.0)
    normalized_vh = np.clip((vh_db + 22.0) / 8.0, -5.0, 5.0)
    normalized = np.stack((normalized_vv, normalized_vh), axis=0)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class EmptyRasterError(ValueError):
    """A crop contains no usable finite non-zero VV/VH signal."""


def sentinel1_water_mask(
    vv: np.ndarray,
    vh: np.ndarray,
    valid_mask: np.ndarray | None = None,
    vv_threshold_db: float = VV_WATER_MAX_DB,
    vh_threshold_db: float = VH_WATER_MAX_DB,
    min_polarization_difference_db: float = VV_VH_MIN_DIFFERENCE_DB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Require low VV, low VH, and a VV/VH contrast before morphology cleanup."""
    vv = np.asarray(vv, dtype=np.float32)
    vh = np.asarray(vh, dtype=np.float32)
    valid = np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    vv_db = 10.0 * np.log10(np.clip(vv, 1e-6, 1.0))
    vh_db = 10.0 * np.log10(np.clip(vh, 1e-6, 1.0))
    water_mask = (
        (vv_db < vv_threshold_db)
        & (vh_db < vh_threshold_db)
        & ((vv_db - vh_db) > min_polarization_difference_db)
        & valid
    )
    footprint = np.ones((MORPHOLOGY_SIZE, MORPHOLOGY_SIZE), dtype=bool)
    water_mask = median_filter(water_mask.astype(np.uint8), size=MORPHOLOGY_SIZE) > 0
    water_mask = binary_closing(water_mask, structure=footprint)
    water_mask = binary_opening(water_mask, structure=footprint)
    water_mask &= valid
    return water_mask.astype(bool), vv_db, vh_db


def water_polygon_from_tiff(
    tiff_path: str | Path,
    model: Any = None,
    device: Any = None,
    mode: str = "sar_threshold",
) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    """Segment one crop and return water geometry, CRS, binary mask, and profile."""
    with rasterio.open(tiff_path) as source:
        if source.count < 2:
            raise ValueError(f"{tiff_path} must contain VV and VH bands.")
        if source.crs is None:
            raise ValueError(f"{tiff_path} has no CRS metadata.")
        vv_band = source.read(1, out_dtype="float32", masked=True)
        vh_band = source.read(2, out_dtype="float32", masked=True)
        vv = np.asarray(vv_band.filled(np.nan), dtype=np.float32)
        vh = np.asarray(vh_band.filled(np.nan), dtype=np.float32)
        valid_pixels = (
            ~np.ma.getmaskarray(vv_band)
            & ~np.ma.getmaskarray(vh_band)
            & np.isfinite(vv)
            & np.isfinite(vh)
            & (vv > 0)
            & (vh > 0)
        )
        if not np.any(valid_pixels):
            raise EmptyRasterError(f"{tiff_path} has no finite, non-zero VV/VH pixels.")

        if mode == "sar_threshold":
            water_mask, _vv_db, _vh_db = sentinel1_water_mask(vv, vh, valid_pixels)
        elif mode == "unet":
            if model is None or device is None:
                raise ValueError("The U-Net mode requires an explicitly loaded trained model.")
            import torch

            normalized = normalize_vv_vh(vv, vh)
            tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)
            with torch.inference_mode():
                prediction = model(tensor).squeeze().detach().cpu().numpy()
            water_mask = (prediction > 0.5) & valid_pixels
        else:
            raise ValueError(f"Unknown inference mode: {mode}")

        raster_transform = source.transform
        raster_crs = source.crs
        raster_profile = source.profile.copy()

    feature_shapes = [
        shape(geometry)
        for geometry, value in shapes(
            water_mask.astype(np.uint8), mask=water_mask, transform=raster_transform
        )
        if int(value) == 1
    ]
    if not feature_shapes:
        return GeometryCollection(), raster_crs, water_mask, raster_profile

    water_gdf = gpd.GeoDataFrame(geometry=feature_shapes, crs=raster_crs).to_crs(
        "EPSG:4326"
    )
    metric_water_gdf = water_gdf.to_crs(METRIC_CRS)
    water_gdf = water_gdf.loc[metric_water_gdf.area >= MIN_WATER_POLYGON_AREA_M2]
    if water_gdf.empty:
        return GeometryCollection(), raster_crs, water_mask, raster_profile
    water_geometry = unary_union(water_gdf.geometry)
    return water_geometry, raster_crs, water_mask, raster_profile


def _graph_edges_gdf(graph: Any) -> gpd.GeoDataFrame:
    edges = ox.graph_to_gdfs(graph, nodes=False)
    if "edge_id" not in edges.columns:
        raise ValueError("OSM graph edges are missing the required edge_id attribute.")
    edges = edges.copy()
    edges["edge_id"] = edges["edge_id"].astype(int)
    if edges.crs is None:
        raise ValueError("OSM graph has no CRS metadata.")
    if str(edges.crs).upper() != "EPSG:4326":
        edges = edges.to_crs("EPSG:4326")
    return edges


def _is_bridge(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_is_bridge(item) for item in value)
    if isinstance(value, str):
        return value.strip().casefold() in {"yes", "true", "1"}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return False


def _flooded_edge_ids_from_intersections(
    edges: gpd.GeoDataFrame, water_polygon: Any
) -> set[int]:
    """Require meaningful metric overlap and never block OSM-tagged bridges."""
    if edges.crs is None:
        raise ValueError("Road edge GeoDataFrame has no CRS metadata.")
    metric_edges = edges.to_crs(METRIC_CRS)
    metric_water = gpd.GeoSeries([water_polygon], crs="EPSG:4326").to_crs(
        METRIC_CRS
    ).iloc[0]
    intersection_lengths_m = metric_edges.geometry.intersection(metric_water).length
    edge_lengths_m = metric_edges.geometry.length
    if "bridge" in edges.columns:
        bridge_mask = edges["bridge"].map(_is_bridge)
    else:
        bridge_mask = pd.Series(False, index=edges.index)
    flooded_mask = (
        (
            intersection_lengths_m + METRIC_LENGTH_TOLERANCE_M
            >= edge_lengths_m * MIN_EDGE_FLOOD_FRACTION
        )
        | (intersection_lengths_m > MIN_EDGE_FLOOD_LENGTH_M + METRIC_LENGTH_TOLERANCE_M)
    ) & ~bridge_mask
    return set(edges.loc[flooded_mask, "edge_id"].astype(int).tolist())


def _save_results(
    date_str: str,
    flooded_edge_ids: set[int],
    water_geometry: Any,
    db_path: str | Path,
) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if water_geometry.is_empty:
        geojson = {"type": "FeatureCollection", "features": []}
    else:
        from shapely.geometry import mapping

        geojson = {
            "type": "Feature",
            "geometry": mapping(water_geometry),
            "properties": {
                "source": "sentinel-1-sar-dual-polarization",
                "vv_max_db": VV_WATER_MAX_DB,
                "vh_max_db": VH_WATER_MAX_DB,
                "vv_vh_min_difference_db": VV_VH_MIN_DIFFERENCE_DB,
            },
        }

    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS daily_flooded_edges "
            "(date TEXT NOT NULL, edge_id INTEGER NOT NULL, "
            "PRIMARY KEY (date, edge_id))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS flood_polygons "
            "(date TEXT PRIMARY KEY, geojson TEXT NOT NULL)"
        )
        connection.execute("DELETE FROM daily_flooded_edges WHERE date = ?", (date_str,))
        connection.executemany(
            "INSERT OR IGNORE INTO daily_flooded_edges (date, edge_id) VALUES (?, ?)",
            [(date_str, edge_id) for edge_id in sorted(flooded_edge_ids)],
        )
        connection.execute(
            "INSERT OR REPLACE INTO flood_polygons (date, geojson) VALUES (?, ?)",
            (date_str, json.dumps(geojson, separators=(",", ":"))),
        )
        connection.commit()


def _save_binary_mask(
    date_str: str,
    water_mask: np.ndarray,
    source_profile: dict[str, Any],
    mask_dir: str | Path = MASK_DIR,
) -> Path:
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    output_path = mask_dir / f"watermask_{date_str}.tif"
    profile = source_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    profile.pop("photometric", None)
    profile.pop("interleave", None)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(water_mask.astype(np.uint8), 1)
    return output_path


def reset_results_database(db_path: str | Path = DB_PATH) -> None:
    """Clear previous detections before regenerating from the current TIFF set."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS daily_flooded_edges "
            "(date TEXT NOT NULL, edge_id INTEGER NOT NULL, "
            "PRIMARY KEY (date, edge_id))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS flood_polygons "
            "(date TEXT PRIMARY KEY, geojson TEXT NOT NULL)"
        )
        connection.execute("DELETE FROM daily_flooded_edges")
        connection.execute("DELETE FROM flood_polygons")
        connection.commit()


def process_geotiff(
    tiff_path: str | Path,
    model: Any,
    device: Any,
    graph: Any,
    db_path: str | Path = DB_PATH,
    mode: str = "sar_threshold",
    mask_dir: str | Path = MASK_DIR,
) -> dict[str, Any]:
    """Infer water, intersect roads, and persist one scene's results."""
    tiff_path = Path(tiff_path)
    match = DATE_PATTERN.match(tiff_path.stem)
    if not match:
        raise ValueError(f"Expected crop filename s1_YYYY-MM-DD.tif, got {tiff_path.name}")
    date_str = date.fromisoformat(match.group(1)).isoformat()
    water_polygon, _raster_crs, water_mask, raster_profile = water_polygon_from_tiff(
        tiff_path, model, device, mode
    )
    _save_binary_mask(date_str, water_mask, raster_profile, mask_dir)
    edges = _graph_edges_gdf(graph)
    if water_polygon.is_empty:
        flooded_edge_ids: set[int] = set()
        water_area_km2 = 0.0
    else:
        metric_edges = edges.to_crs(METRIC_CRS)
        metric_water = gpd.GeoSeries([water_polygon], crs="EPSG:4326").to_crs(
            METRIC_CRS
        ).iloc[0]
        flooded_edge_ids = _flooded_edge_ids_from_intersections(edges, water_polygon)
        water_area_km2 = float(
            gpd.GeoSeries([water_polygon], crs="EPSG:4326").to_crs(METRIC_CRS)
            .area.iloc[0]
            / 1_000_000
        )
    _save_results(date_str, flooded_edge_ids, water_polygon, db_path)
    return {
        "date": date_str,
        "flooded_edges": len(flooded_edge_ids),
        "water_area_km2": water_area_km2,
        "mode": mode,
    }


def run_batch(
    tiff_dir: str | Path = TIFF_DIR,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    weights_path: str | Path = WEIGHTS_PATH,
    db_path: str | Path = DB_PATH,
    device: str | None = None,
    mode: str = "sar_threshold",
    mask_dir: str | Path = MASK_DIR,
) -> list[dict[str, Any]]:
    """Rebuild detections from the current downloaded crop set."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tiff_dir = Path(tiff_dir)
    if not tiff_dir.is_dir():
        LOGGER.warning("TIFF directory does not exist: %s", tiff_dir)
        return []
    tiff_paths = sorted(tiff_dir.glob("*.tif"))
    if not tiff_paths:
        LOGGER.warning("No Sentinel-1 TIFF crops found in %s; database not changed.", tiff_dir)
        return []
    graph_path = Path(graph_path)
    graph = (
        ox.load_graphml(filepath=graph_path)
        if graph_path.exists()
        else build_road_graph(graph_path)
    )
    if mode == "unet":
        model, selected_device = load_model(weights_path, device)
    elif mode == "sar_threshold":
        model, selected_device = None, None
        LOGGER.info("Using VV dB threshold detector (< %.1f dB).", WATER_THRESHOLD_DB)
    else:
        raise ValueError("mode must be 'sar_threshold' or 'unet'.")

    reset_results_database(db_path)
    results = []
    for tiff_path in tiff_paths:
        try:
            result = process_geotiff(
                tiff_path,
                model,
                selected_device,
                graph,
                db_path,
                mode=mode,
                mask_dir=mask_dir,
            )
        except EmptyRasterError as error:
            LOGGER.warning("Skipping empty crop %s: %s", tiff_path.name, error)
            continue
        except Exception:
            LOGGER.exception("Inference failed for %s", tiff_path.name)
            continue
        LOGGER.info(
            "%s: %d intersecting road edges, water area %.2f km2",
            result["date"],
            result["flooded_edges"],
            result["water_area_km2"],
        )
        results.append(result)
    return results


def main() -> None:
    run_batch(mode="sar_threshold")


if __name__ == "__main__":
    main()