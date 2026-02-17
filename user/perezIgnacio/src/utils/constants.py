# AgriPotential Dataset URL
DATASET_URL = "https://huggingface.co/datasets/m-sakka/agripotential/resolve/main/"

# Sentinel-2 band names
BAND_NAMES = [
    "B1",   # Coastal Aerosol
    "B2",   # Blue
    "B3",   # Green
    "B4",   # Red
    "B5",   # Vegetation Red Edge 1
    "B6",   # Vegetation Red Edge 2
    "B7",   # Vegetation Red Edge 3
    "B8",   # NIR (Near Infrared)
    "B8A",  # Narrow NIR
    "B9",   # Water vapour
    "B11",  # SWIR 1 (Short-wave Infrared 1)
    "B12"   # SWIR 2 (Short-wave Infrared 2)
]

# Class labels for agricultural potential
POTENTIAL_CLASSES = [
    "unlabelled",
    "very low",
    "low", 
    "average",
    "high",
    "very high"
]

# Map class IDs to names (0 = unlabelled, 1-5 = potential levels)
ID_TO_CLASS = {i: name for i, name in enumerate(POTENTIAL_CLASSES)}

# Normalization constants for Sentinel-2
SENTINEL2_MAX = 10000.0  # Reflectance values are 0-10000

# Data subsets
SUBSETS = ["train", "val", "test"]
