# Data preparation

The source datasets are not included in this repository.

## ERA5

**Dataset:** ERA5 hourly data on single levels from 1940 to present  
**Provider:** Copernicus Climate Change Service (C3S), Climate Data Store  
**Official dataset page:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels  
**DOI:** https://doi.org/10.24381/cds.adbb2d47

### Required variables

- `u10`: 10-m u-component of wind
- `v10`: 10-m v-component of wind

The preprocessing script expects a NetCDF file with `time`, `latitude`, `longitude`, `u10`, and `v10`.

### Range used in the experiment

- Time: 2014-01-01 00:00 to 2014-12-31 23:00, hourly
- Latitude: 14°N to 54°N
- Longitude: 114°E to 154°E
- ERA5 atmospheric resolution: 0.25° × 0.25°
- HR field: 161 × 161
- LR field: 32 × 21

Data split:

- Train: 2014-01-01 00:00 ≤ time < 2014-09-01 00:00
- Validation: 2014-09-01 00:00 ≤ time < 2014-10-01 00:00
- Test: 2014-10-01 00:00 ≤ time < 2015-01-01 00:00

For a complete hourly 2014 file, the split contains 5,832 / 720 / 2,208 fields.

### Preprocessing

`preprocess.py` performs the following steps:

1. Select the configured time interval and geographic domain.
2. Reorder latitude to ascending order when needed.
3. Replace missing values in each wind-component frame with the finite mean of that frame.
4. Compute wind speed from `u10` and `v10`.
5. Generate the LR wind-speed field with box-mean aggregation to 32 × 21.
6. Fit the wind-speed scaler using the training HR fields only.
7. Apply `log1p` scaling and map wind speed to `[-1, 1]`.
8. Scale `u10` and `v10` using the maximum absolute component value from the training split.
9. Save arrays, coordinates, timestamps, split indices, and scaler values in the preprocessing cache.

Default local path:

```text
data/era5_2014.nc
```

## IBTrACS

**Dataset:** International Best Track Archive for Climate Stewardship (IBTrACS)  
**Provider:** NOAA National Centers for Environmental Information (NCEI)  
**Official dataset page:** https://www.ncei.noaa.gov/products/international-best-track-archive  
**Subset used in the manuscript evaluation:** Western Pacific (`WP`) basin  
**Version used in the manuscript evaluation:** v04r00

For the manuscript configuration, use the archived Version 4 Western Pacific CSV:

```text
ibtracs.WP.list.v04r00.csv
```

The evaluator accepts the standard Version 4 column layout and uses timestamp, storm identifier, latitude, longitude, storm name when present, and available wind-intensity columns.

### Event matching

During evaluation:

1. IBTrACS records are restricted to the ERA5 test period and spatial domain.
2. ERA5 frames and IBTrACS records are matched by exact timestamp.
3. If multiple in-domain storms occur at the same timestamp, the strongest available record is retained for that frame.
4. The event region is determined from the HR reference field.
5. The spectral metric is computed over the matched event frames.

Default local path:

```text
data/ibtracs.WP.list.v04r00.csv
```

Data licensing and citation requirements remain those of the original providers.
