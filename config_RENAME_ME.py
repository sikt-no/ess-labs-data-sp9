"""
Config file for ess-eosc

You must rename config_RENAME_ME.py to config.py!

Do NOT commit your config.py to git!
"""

from pathlib import Path

# Paths for download and output data
EEA_PATH = Path(<my_path>)
ERA5_PATH = Path(<my_path>)

# Folder for temporary files
TMP_PATH = Path(<my_path>)

# path to local copy of population data file.
# you DO need a local copy before running the code.
GLOBAL_POP_FILE = Path(<my_path>)

# path where the NUTS regions files will be stored locally after download.
# you don't need to download a local copy of these before running the code.
NUTS_PATH = Path(<my_path>)
