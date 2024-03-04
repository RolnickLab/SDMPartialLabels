This document explains how the dataset was generated starting from the raw data:

Dataset Raw and final, with all intermediate files is in: `/network/projects/ecosystem-embeddings/SatButterfly_dataset`

# Raw Data:
* Downloaded raw data from [eButterfly Surveys ](https://www.gbif.org/dataset/cf3bdc30-370c-48d3-8fff-b587a39d72d6)
* DARWIN CORE ARCHIVE includes all the observations around the world. Download was done in May 2023.
* Data exists in `Darwin/0177350-230224095556074/occurrence.txt`, file has been reformatted to csv and saved in `SatButterfly_dataset/occurrence_.csv`
* Filter observations to consider only observations in the USA starting the year 2010 or later: `SatButterfly_dataset/occ_usa.csv`
* Filter out observations outside the continental USA following [this](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html): `cb_2018_us_nation_5m/cb_2018_us_nation_5m.shp`, resulting in `SatButterfly_dataset/occ_usa_geo_filtered.csv`. This also excludes locations in Alaska and Hawaii.
* Then, we prepare two versions of the dataset explained below.

## Unifying species between SatButterfly-v1 and SatButterfly-v2:
Species observed were different between SatButterfly-v1 and SatButterfly-v2. So, after clustering and preparing final hotspots, and before preparing final targets, we took the intersection of the 2 species lists and ended up with a unified species list for butterflies. We also filtered out species with frequency less than 100, resulting in 172 species.

Final species list contains: species that have appeared at least 100 times across the dataset, and appeared in both versions of the dataset

- `intersection_species_list.txt`: 601 species
- `intersection_species_list_frequency_ge100.txt`: 172 species
- `union_species_list.txt`: all species that occur in both versions
- `union_species_list_frequency_ge100.txt`: all species that occur in both versions, with frequenecy >= 100

# Data preparation:
## SatButterfly-v1:
Notebook `prepare_ebutterfly_independent_from_ebird.ipynb` includes all steps to prepare the final data.
- We first cluster ebutterfly observations with DBSCAN.
- We ended-up with 7608 hotspots, also split using DBSCAN, following SatBird, resulting in: ('valid', 1147), ('test', 1145), ('train', 5316),

#### Targets:
ebutterfly checklists are aggregated per hotspot over the years 2010 to 2023. Butterfly targets have 2 versions, all butterfly's 601 species or 172 species (that have frequency >= 100).
- butterfly_targets_v1.1 : 601 species
- butterfly_targets_v1.2 : 172 species (used in our final experiments)

#### Satellite images:
To generate satellite images, we followed these steps:
- Create polygons around the latitudes and longitudes after clustering the data
- Use polygons to extract satellite images: a) RGBNIR (["B02", "B03", "B04", "B08"]), and b) RGB: [“visual”]) using Planetary Computer API, script in the repo: `data_processing/ebutterfly_data_preparation/download_rasters_from_planetary_computer.py`
- Excludes hotspots with satellite images if the size of the image is less than 128x128.

#### Env data:
Following SatBird, we use the polygons file to extract environmental rasters, using the script `data_processing/environmental/get_env_var.py`

#### Species:
- `species/full_species_list.csv`: full species list
- `species/species_list_frequency_ge100.csv`: species list with observed >= 100 times
- `species/species_list_updated_601species.csv`: species list aligned with SatButterfly-v2
- `species/species_list_updated_172species.csv`: species list aligned with SatButterfly-v2 with species observed >= 100 times

#### Final data used for training:
- `butterfly_hotspots_train.csv`
- `butterfly_hotspots_valid.csv`
- `butterfly_hotspots_test.csv`
- butterfly_targets_v1.2/
- environmental/
- images/
- images_visual/

## SatButterfly-v2:
Notebook `prepare_ebutterfly_with_ebird.ipynb` includes all steps to prepare the final data. We collocate butterfly observations with SatBird-USA-summer hotspots, so we can have hotspots where both birds and butterflies are observed.
* We perform BallTree-based KNN clustering where SatBird hotspots are used as centroids. 
* We then search within $1$ km for neighbour butterfly observations using haversine distance. 
* Finally, butterfly targets are aggregated and recorded for SatBird hotspots wherever available. 
* We end up with available butterfly targets in a small subset of SatBird's train/validation/test splits: ('valid', 1076), ('test', 958), ('train', 4650)
* There is no need to download satellite images or env data as it is in the same locations as SatBird.

#### Targets:
ebutterfly targets are aggregated per hotspot, over the years 2010 to 2023.
- butterfly_targets_v1.1 : 601 species
- butterfly_targets_v1.2 : 172 species (used in our final experiments)

#### Species:
- `species/full_species_list.csv`: full species list
- `species/species_list_frequency_ge100.csv`: species list with observed >= 100 times
- `species/species_list_updated_601species.csv`: species list aligned with SatButterfly-v2
- `species/species_list_updated_172species.csv`: species list aligned with SatButterfly-v2 with species observed >= 100 times

#### Final data used for training:
- `butterfly_hotspots_ebird_train.csv`
- `butterfly_hotspots_ebird_valid.csv`
- `butterfly_hotspots_ebird_test.csv`
- butterfly_targets_v1.2/
