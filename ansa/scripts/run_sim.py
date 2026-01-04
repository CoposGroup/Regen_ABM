import _setup_path
import importlib
import abm, config
from utils.profiler import profiling

config.OUTPUT_DIR = f'sim_output'

importlib.reload(abm)
data_dict = abm.run_simulation()