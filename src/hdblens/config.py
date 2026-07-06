"""Central configuration: paths, dataset IDs, and geospatial anchors."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"

RAW_CSV = DATA_DIR / "resale_2017.csv"

# ---------------------------------------------------------------------------
# data.gov.sg dataset: "Resale flat prices based on registration date
# from Jan-2017 onwards" (HDB, updated monthly)
# ---------------------------------------------------------------------------
DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
API_BASE = "https://api-open.data.gov.sg/v1/public/api/datasets"

# ---------------------------------------------------------------------------
# Geospatial anchors (approximate town centroids, WGS84)
# ---------------------------------------------------------------------------
TOWN_COORDS: dict[str, tuple[float, float]] = {
    "ANG MO KIO": (1.3691, 103.8454),
    "BEDOK": (1.3236, 103.9273),
    "BISHAN": (1.3526, 103.8352),
    "BUKIT BATOK": (1.3590, 103.7637),
    "BUKIT MERAH": (1.2819, 103.8239),
    "BUKIT PANJANG": (1.3774, 103.7719),
    "BUKIT TIMAH": (1.3294, 103.8021),
    "CENTRAL AREA": (1.2897, 103.8501),
    "CHOA CHU KANG": (1.3840, 103.7470),
    "CLEMENTI": (1.3162, 103.7649),
    "GEYLANG": (1.3201, 103.8918),
    "HOUGANG": (1.3612, 103.8863),
    "JURONG EAST": (1.3329, 103.7436),
    "JURONG WEST": (1.3404, 103.7090),
    "KALLANG/WHAMPOA": (1.3100, 103.8651),
    "MARINE PARADE": (1.3020, 103.8971),
    "PASIR RIS": (1.3721, 103.9474),
    "PUNGGOL": (1.3984, 103.9072),
    "QUEENSTOWN": (1.2942, 103.7861),
    "SEMBAWANG": (1.4491, 103.8185),
    "SENGKANG": (1.3868, 103.8914),
    "SERANGOON": (1.3554, 103.8679),
    "TAMPINES": (1.3496, 103.9568),
    "TOA PAYOH": (1.3343, 103.8563),
    "WOODLANDS": (1.4382, 103.7890),
    "YISHUN": (1.4304, 103.8354),
}

# Raffles Place — proxy for the CBD
CBD = (1.2840, 103.8515)

# URA-designated regional centres (decentralised employment hubs)
REGIONAL_CENTRES: dict[str, tuple[float, float]] = {
    "Tampines": (1.3530, 103.9450),
    "Jurong Lake District": (1.3331, 103.7422),
    "Woodlands": (1.4360, 103.7860),
    "Punggol Digital District": (1.4052, 103.9024),
}

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TARGET = "resale_price"
QUANTILES = (0.10, 0.50, 0.90)

CATEGORICAL_FEATURES = ["town", "flat_type", "flat_model"]
NUMERIC_FEATURES = [
    "floor_area_sqm",
    "storey_mid",
    "remaining_lease_months",
    "flat_age_years",
    "dist_cbd_km",
    "dist_regional_centre_km",
    "month_index",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Time-based split boundaries (leakage-safe evaluation)
TRAIN_END = "2024-12"   # train: 2017-01 .. 2024-12
VAL_END = "2025-12"     # val:   2025-01 .. 2025-12
                        # test:  2026-01 .. latest
