This repository contains the code to download data for Climate Neutral and Smart Cities, the 9th of EOSC Future Task 6.3’s Science Project.

## Setup:

- Python must be installed. The code has been tested with Python 3.9.
- Python requirements are listed in the file requirements.txt. They can be installed using `pip install -r requirements.txt`.
- To download ERA5 data from the Copernicus Climate Data Store, an API key is needed. It can be obtained by following instructions here: https://cds.climate.copernicus.eu/api-how-to.
- A file containing the GHS population grid must be downloaded from here: https://ghsl.jrc.ec.europa.eu/download.php?ds=pop. Tested with 1km resolution and Mollweide coordinate system.
- SPSS files containing ESS data must be downloaded: ESS8e02_2.sav, ESS9e03_1.sav, ESS10.sav, ESS10SC.sav
- The file config_RENAME_ME.py must be populated with local paths and renamed to config.py.

## Files

- era5-download.py: Downloads ERA5 raw data
- era5-prepare.py: Prepares ERA5 data for merging with ESS
- eea-download.py: Downloads EEA (European Environment Agency) raw data
- eea-prepare.py: Prepares EEA data for merging with ESS
- merge.py: Merges prepared data with ESS files.
