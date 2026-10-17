import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import os
import glob
from tqdm import tqdm
import warnings
import pylandstats as pls
import gc
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

print("8-Category Analysis (7 Main + 1 'Others')")

# Directories
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Input files
STATIONS_CSV = 'mekong_delta_water_quality_1985_2023.csv'
LULC_RASTERS_DIR = 'MekongDelta_VLUCD_1990_2020'

# Buffer distances (km) - 10, 15, 20, 25 km for sensitivity analysis
BUFFER_DISTANCES_KM = [1,2,3,4,5,6,7,8,9,10,11,12,13,14, 15,16,17,18,19, 20,21,22,23,24, 25,26,28,29,30]

# Using Level-2 but follow JAXA VLUCD Level-1 Official Classification:
#   #1 and #2: Residential land
#   #3: Rice paddies
#   #4, #5, and #6: Croplands
#   #7: Grassland
#   #8: Barren land
#   #9: Scrub/Shrub
#   #10, #11, #12, #14, and #20: Forests
#   #15: Wetlands (Mangrove!)
#   #16: (Inland wetland - part of Wetlands)
#   #18: Open water
#   #19: Aquaculture

LULC_CATEGORIES = {
    'Residential_Urban': [1, 2],           # L1: Residential land
    'Rice_Cultivation': [3],               # L1: Rice paddies
    'Other_Crops': [4, 5, 6],              # L1: Croplands
    'Aquaculture': [19],                   # L1: Aquaculture
    'Forest': [10, 11, 12, 14, 20],        # L1: Forests (NO Mangrove!)
    'Wetland': [15, 16],                   # L1: Wetlands (Mangrove + Inland)
    'Water_Bodies': [18]                   # L1: Open water
}

# Others will capture: 7 (Grassland), 8 (Barren), 9 (Scrub)

# Coordinate systems
STATION_CRS = 'EPSG:4326'
BUFFER_CRS = 'EPSG:32648'
N_JOBS = -1  # Use all available CPU cores

print(f"\n CONFIGURATION:")
print(f"   Stations file: {STATIONS_CSV}")
print(f"   LULC rasters directory: {LULC_RASTERS_DIR}")
print(f"   Buffer distances: {BUFFER_DISTANCES_KM} km")
print(f"   LULC categories: 7 Main + 1 'Others' (Grassland, Barren, Scrub)")
print(f"   Parallel Cores (n_jobs): {N_JOBS} (all available)")
print(f"   Output directory: {CACHE_DIR}/")

print("\n JAXA-ALIGNED CATEGORY MAPPING:")
for cat, codes in LULC_CATEGORIES.items():
    print(f"   {cat}: {codes}")
print(f"   Others: [7, 8, 9] (auto-computed)")


def calculate_metrics_fast(ls, class_codes, landscape_area_ha):
    """Calculate landscape configuration metrics for merged LULC category."""
    try:
        if not any(c in ls.classes for c in class_codes):
            return {'PD': 0, 'ED': 0, 'LSI': 1, 'LPI': 0, 'num_patches': 0, 'total_area_ha': 0}
        
        total_patches, total_area_m2, total_edge, max_patch_area = 0, 0, 0, 0
        
        for code in class_codes:
            if code not in ls.classes:
                continue
            try:
                total_patches += ls.number_of_patches(class_val=code)
                total_area_m2 += ls.total_area(class_val=code)
                try:
                    total_edge += ls.total_edge(class_val=code, count_boundary=True)
                except:
                    pass
                try:
                    patch_areas = ls.area(class_val=code)
                    if len(patch_areas) > 0:
                        max_patch_area = max(max_patch_area, patch_areas.max())
                except:
                    pass
            except:
                continue
        
        total_area_ha = total_area_m2 / 10000
        pd_value = (total_patches / landscape_area_ha) * 100 if landscape_area_ha > 0 else 0
        ed_value = total_edge / landscape_area_ha if landscape_area_ha > 0 else 0
        lpi_value = (max_patch_area / 10000 / landscape_area_ha * 100) if landscape_area_ha > 0 else 0
        min_edge = 4 * np.sqrt(total_area_m2) if total_area_m2 > 0 else 0
        lsi_value = total_edge / min_edge if min_edge > 0 else 1
        
        return {'PD': pd_value, 'ED': ed_value, 'LSI': lsi_value, 'LPI': lpi_value,
                'num_patches': total_patches, 'total_area_ha': total_area_ha}
    except Exception:
        return {'PD': np.nan, 'ED': np.nan, 'LSI': np.nan, 'LPI': np.nan,
                'num_patches': np.nan, 'total_area_ha': np.nan}

def calculate_buffer_metrics(src, buffer_geom_raster_crs, geometry_info):
    """Extract raster data within buffer and compute all LULC metrics."""
    try:
        out_image, _ = mask(src, [mapping(buffer_geom_raster_crs)], crop=True)
        clipped_array = out_image[0]
        
        valid_mask = (clipped_array > 0) & (clipped_array < 255)
        if not np.any(valid_mask):
            return None
        
        clipped_clean = np.where(valid_mask, clipped_array, 0)
        
        pixel_area_m2 = src.res[0] * src.res[1]
        landscape_area_ha = (valid_mask.sum() * pixel_area_m2) / 10000
        
        result = geometry_info.copy()
        result['landscape_area_ha'] = landscape_area_ha
        
        unique_vals, counts = np.unique(clipped_array[valid_mask], return_counts=True)
        pixel_counts = dict(zip(unique_vals, counts))
        total_valid_pixels = valid_mask.sum()
        
        ls = pls.Landscape(clipped_clean, res=src.res, nodata=0)
        
        # Process the 7 main categories
        composition_sum = 0
        for cat_name, codes in LULC_CATEGORIES.items():
            pixel_sum = sum(pixel_counts.get(code, 0) for code in codes)
            pct = (pixel_sum / total_valid_pixels) * 100
            result[f'{cat_name}_composition_pct'] = pct
            composition_sum += pct
            
            metrics = calculate_metrics_fast(ls, codes, landscape_area_ha)
            result[f'{cat_name}_config_PD'] = metrics['PD']
            result[f'{cat_name}_config_ED'] = metrics['ED']
            result[f'{cat_name}_config_LSI'] = metrics['LSI']
            result[f'{cat_name}_config_LPI'] = metrics['LPI']
        
        # Others = Grassland (7) + Barren (8) + Scrub (9)
        result['Others_composition_pct'] = 100.0 - composition_sum
        result['Others_config_PD'] = 0.0
        result['Others_config_ED'] = 0.0
        result['Others_config_LSI'] = 1.0
        result['Others_config_LPI'] = 0.0
        
        return result
        
    except Exception as e:
        print(f"  Error processing buffer: {e}")
        return None

def process_one_station(station_row, raster_filepath, buffer_radius_m, buffer_km, year, buffer_crs):
    """Worker function to process a single station."""
    _, station = station_row
    try:
        with rasterio.open(raster_filepath) as src:
            buffer_geom = station.geometry.buffer(buffer_radius_m)
            buffer_gdf = gpd.GeoDataFrame([{'geometry': buffer_geom}], crs=buffer_crs)
            buffer_geom_raster_crs = buffer_gdf.to_crs(src.crs).iloc[0].geometry
            
            info = {
                'year': year,
                'stationName': station['stationName'],
                'River': station['River'],
                'buffer_km': buffer_km
            }
            
            result = calculate_buffer_metrics(src, buffer_geom_raster_crs, info)
            return result
        
    except Exception as e:
        print(f" Failed processing {station['stationName']} for {year}: {e}")
        return None

if __name__ == "__main__":
    try:        
        stations_df = pd.read_csv(STATIONS_CSV)
        unique_stations = stations_df.drop_duplicates(subset='stationName').reset_index(drop=True)
        required_cols = ['stationName', 'longitude', 'latitude', 'River']
        if any(col not in unique_stations.columns for col in required_cols):
            raise ValueError("Missing required columns")
            
        stations_gdf = gpd.GeoDataFrame(
            unique_stations,
            geometry=gpd.points_from_xy(unique_stations['longitude'], unique_stations['latitude']),
            crs=STATION_CRS
        )
        stations_gdf_metric = stations_gdf.to_crs(BUFFER_CRS)
        print(f" Loaded {len(stations_gdf)} unique stations")

        
        lulc_files = sorted(glob.glob(os.path.join(LULC_RASTERS_DIR, '*.tif')))
        if not lulc_files:
            raise FileNotFoundError(f"No LULC raster files found in '{LULC_RASTERS_DIR}'")
            
        year_file_map = {}
        for filepath in lulc_files:
            year_str = os.path.basename(filepath).split('_')[-1].replace('.tif', '')
            if year_str.isdigit():
                year_file_map[int(year_str)] = filepath
        print(f" Loaded {len(year_file_map)} LULC raster files")
        print(f"PROCESSING BUFFERS {BUFFER_DISTANCES_KM} km (JAXA-ALIGNED)")
        
        for buffer_km in BUFFER_DISTANCES_KM:
            buffer_radius_m = buffer_km * 1000
            output_file = os.path.join(CACHE_DIR, f"lulc_metrics_corrected_{buffer_km}km.csv")
            
            # Check if file exists to skip
            if os.path.exists(output_file):
                print(f"✓ [{buffer_km:2d} km] Already exists: {output_file}")
                continue

            print(f"Processing {buffer_km} km buffer (JAXA-aligned categories)")
            
            buffer_results = []
            
            for year in tqdm(sorted(year_file_map.keys()), desc=f"Years ({buffer_km} km)"):
                raster_filepath = year_file_map[year]
                
                job_list = [
                    delayed(process_one_station)(
                        station_row, raster_filepath, buffer_radius_m, buffer_km, year, BUFFER_CRS
                    ) 
                    for station_row in stations_gdf_metric.iterrows()
                ]
                
                station_results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(job_list)
                buffer_results.extend([r for r in station_results if r is not None])
                gc.collect()
            
            if buffer_results:
                results_df = pd.DataFrame(buffer_results)
                results_df.to_csv(output_file, index=False)
                print(f"   Saved {len(results_df):,} records → {output_file}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
