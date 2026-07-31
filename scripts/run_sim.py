import _setup_path
import importlib
import abm, config
from utils.profiler import profiling

# Example parameters. All other parameters default to those in src/config.py
config.MIGRATION_ENABLED = True
config.SOFTENING_ENABLED = True
config.SOFTENING_SWAP_TIME = 2.0
config.MIGRATION_FRACTION = 0.5

importlib.reload(abm)
abm.OUTPUT_DIR = 'mig50_swap'
data_dict = abm.run_simulation()