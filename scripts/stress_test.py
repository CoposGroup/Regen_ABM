"""Stress test (poking) simulations for limb regeneration. 

to-do: 
- now that we have viscoelasticity, i need to make sure the rest lengths are properly updated when pulling the snapshots
- viscoelasticity should probably then be turned off, as AFM experiments don't take 2 days (should think about the AFM timescale)
"""
import _setup_path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import importlib
import multiprocessing as mp
from config import *
from abm import tune_F_per_segment
import datetime
import pickle

date_str = datetime.datetime.now().strftime('%m-%d-%y_%H%M')
data_dir = 'Cases2026-01-02/SOFT/MIGRATION30'

out_dir = f'StressTesting/{date_str}'

def get_data(t=1, soft=True, data_dir_local=None):
    """Load boundary, cell and rest length data at specified time."""
    if data_dir_local is None:
        data_dir_local = data_dir
    ###
    blp0 = None
    ###
    try:
        with open(f'{data_dir_local}/data_dict.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        times = data_dict['times']
        idx_t = np.argmin(np.abs(times - t))  # Find closest time in case of floating point errors
        Xe = data_dict['boundaries'][idx_t]
        pos = data_dict['positions'][idx_t]
        ###
        if 'rest_lengths' in data_dict and len(data_dict['rest_lengths']) > idx_t:
            blp0 = data_dict['rest_lengths'][idx_t]
        ###
    except FileNotFoundError:
        print('data_dict.pkl not found, looking for boundary_time_series.csv...')
        Xe_df = pd.read_csv(os.path.join(data_dir_local, 'boundary_time_series.csv'))
        Xe = Xe_df.loc[np.abs(Xe_df['time'] - t) < 1e-10][['x', 'y']].values
        pos_df = pd.read_csv(os.path.join(data_dir_local, 'cells.csv'))
        pos = pos_df.loc[np.abs(pos_df['time'] - t) < 1e-10][['x', 'y']].values

    return Xe, pos, len(pos), blp0

def run_stress_test(t0, force_app_time=2.0, soft=True, data_dir_local=None, out_dir_local=None):
    """Run stress test for a single time point."""
    import config
    import sys
    
    if data_dir_local is None:
        data_dir_local = data_dir
    if out_dir_local is None:
        out_dir_local = out_dir
    
    config.VIDEO_FLAG = True
    config.PROFILING_FLAG = False  
    config.MIGRATION_ENABLED = False
    config.INTERCALATION_ENABLED = False
    config.JAMMING_ENABLED = False
    config.EXT_STRESS_FORCE = True
    config.EXT_FORCE_DELAY = 1
    config.TMAX = force_app_time + config.EXT_FORCE_DELAY
    config.OUTPUT_DIR = f'{out_dir_local}/{t0}Days'
    config.ALLOW_SOFTENING = soft
    config.T_DORMANT = config.TMAX
    config.KDEATH = 0.0
    config.REGULATION_FRONT_FLAG = False
    config.GRADIENT = None
    ###
    config.EPI_TYPE = 'linear'
    ###
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    import abm
    importlib.reload(abm)

    Xe, pos, N0, blp0_snap = get_data(t0, soft=soft, data_dir_local=data_dir_local)

    _, _, _, Db_loaded, blp0, blm0, dsb, kb_vals, _ = abm.build_boundaries(soft=config.ALLOW_SOFTENING)
    ###
    if blp0_snap is not None:
        blp0 = blp0_snap.copy()
        blm0 = np.roll(blp0, 1)
    else:
        print(f"[t={t0}] WARNING: no rest_lengths in snapshot, epithelium will be treated as pre-stressed")
    ###

    force_points = config.POKING_POINTS
    K_EXT_tuned = tune_F_per_segment(Xe, Db_loaded, config.FORCE_PER_UNIT_LENGTH, points=force_points)
    print(f"[t={t0}] K_EXT tuned to: {K_EXT_tuned:.6f}")
    sys.stdout.flush()
    
    abm.K_EXT = K_EXT_tuned
    config.K_EXT = K_EXT_tuned
    abm.Xe = Xe.copy()
    abm.Db = Db_loaded
    abm.blp0 = blp0
    abm.blm0 = blm0
    abm.dsb = dsb
    abm.kb_vals = kb_vals
    abm.N0 = N0
    abm.pos[:N0, :] = pos
    abm.pos[N0:, :] = np.nan
    abm.cycle_phases[:N0] = 0
    abm.cycle_phases[N0:] = -1
    
    print(f'[t={t0}] Starting stress test...')
    sys.stdout.flush()
    
    try:
        data_dict = abm.run_simulation()
        print(f'[t={t0}] Complete!')
        sys.stdout.flush()
        return (t0, data_dict['deformation'], data_dict['final_cell_count'], K_EXT_tuned, True)
    except Exception as e:
        print(f'[t={t0}] ERROR: {e}')
        sys.stdout.flush()
        return (t0, None, None, None, False)

def run_all(times=range(8), force_app_time=2.0, soft=True, parallel=False, max_workers=None):
    """
    Run stress tests for all time points.
    
    Args:
        parallel: If True, run in parallel (default: False)
        max_workers: Max parallel workers (None = all CPU cores)
    """
    print(f"{'='*60}")
    print(f"Running {len(times)} stress tests {'in parallel' if parallel else 'sequentially'}")
    print(f"Output directory: {out_dir}")
    if parallel:
        if max_workers is None:
            print(f"Using all available CPU cores: {mp.cpu_count()}")
        else:
            print(f"Using {max_workers} parallel workers")
    print(f"{'='*60}\n")
    
    jobs = [(t0, force_app_time, soft, data_dir, out_dir) for t0 in times]
    
    if parallel:
        with mp.Pool(processes=max_workers) as pool:
            results_list = pool.starmap(run_stress_test, jobs)
    else:
        results_list = [run_stress_test(*job) for job in jobs]
    
    results = {}
    successful = 0
    for t0, dx, cell_count, K_EXT_tuned, success in results_list:
        if success:
            results[t0] = {'dx': dx, 'cell_count': cell_count, 'K_EXT_tuned': K_EXT_tuned}
            successful += 1
        else:
            results[t0] = {'dx': None, 'cell_count': None, 'K_EXT_tuned': None}
    
    print(f"\n{'='*60}")
    print(f"All stress tests complete!")
    print(f"Successful: {successful}/{len(times)}")
    print(f"{'='*60}\n")
    
    valid_times = [t for t in times if results[t]['dx'] is not None]
    valid_dx = [results[t]['dx'] * CONVERSION_FACTOR_UM for t in valid_times]
    
    if len(valid_times) > 0:
        plt.figure(figsize=(10, 5))
        plt.bar(valid_times, valid_dx)
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Deformation (um)', fontsize=12)
        plt.title('Deformation vs Time', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'deformation_vs_time.pdf'))
        plt.close()
        print(f"Plot saved to: {os.path.join(out_dir, 'deformation_vs_time.pdf')}")
    
    print("\nResults:")
    print(results)
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(out_dir, 'results.csv'))
    print(f"Results saved to: {os.path.join(out_dir, 'results.csv')}")
    
    return results

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    results = run_all(
        times=[0, 1, 2, 3, 4, 5, 6, 7],
        force_app_time=2.0,
        soft=True,
        parallel=False,
        max_workers=1
    )
