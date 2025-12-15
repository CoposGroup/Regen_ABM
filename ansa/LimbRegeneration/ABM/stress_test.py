"""
Stress test (poking) simulations for limb regeneration. 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import multiprocessing as mp
from config import *
from abm11 import tune_F_per_segment
import datetime
import pickle

date_str = datetime.datetime.now().strftime('%m-%d-%y_%H%M')
# data_dir = 'data/output/Cases2025-11-12/SOFT/DEFAULT'
data_dir = 'data/output/migration30_soft' ###

out_dir = f'data/output/StressTesting/{date_str}migration 30' ###

def get_data(t=1, soft=True):
    """Load boundary and cell data at specified time."""
    try:
        Xe_df = pd.read_csv(os.path.join(data_dir, 'boundary_time_series.csv'))
        Xe = Xe_df.loc[np.abs(Xe_df['time'] - t) < 1e-10][['x', 'y']].values
        pos_df = pd.read_csv(os.path.join(data_dir, 'cells.csv'))
        pos = pos_df.loc[np.abs(pos_df['time'] - t) < 1e-10][['x', 'y']].values
    except FileNotFoundError:
        print('using data_dict.pkl...')
        with open(f'{data_dir}/data_dict.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        idx_t = data_dict['times'].tolist().index(t)
        Xe = data_dict['boundaries'][idx_t]
        pos = data_dict['positions'][idx_t]

    return Xe, pos, len(pos)

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
    config.EXT_FORCE_DELAY = max(t0, 3.0) + 3.0 # if t0 is greater than or equal to 3 dpa, let it relax longer
    config.TMAX = force_app_time + config.EXT_FORCE_DELAY
    config.OUTPUT_DIR = f'{out_dir_local}/{t0}Days'
    config.ALLOW_SOFTENING = soft
    config.T_DORMANT = config.TMAX
    config.KDEATH = 0.0
    config.REGULATION_FRONT_FLAG = False
    config.GRADIENT = None
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    import abm11
    importlib.reload(abm11)
    
    Xe, pos, N0 = get_data(t0, soft=soft)
    
    _, _, _, Db_loaded, blp0, blm0, dsb, kb_vals, _ = abm11.build_boundaries(soft=config.ALLOW_SOFTENING)
    
    force_points = np.array([94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105])
    K_EXT_tuned = tune_F_per_segment(Xe, Db_loaded, config.FORCE_PER_UNIT_LENGTH, points=force_points)
    print(f"[t={t0}] K_EXT tuned to: {K_EXT_tuned:.6f}")
    sys.stdout.flush()
    
    abm11.K_EXT = K_EXT_tuned
    config.K_EXT = K_EXT_tuned
    abm11.Xe = Xe.copy()
    abm11.Db = Db_loaded
    abm11.blp0 = blp0
    abm11.blm0 = blm0
    abm11.dsb = dsb
    abm11.kb_vals = kb_vals
    abm11.N0 = N0
    abm11.pos[:N0, :] = pos
    abm11.pos[N0:, :] = np.nan
    abm11.cycle_phases[:N0] = 0
    abm11.cycle_phases[N0:] = -1
    
    print(f'[t={t0}] Starting stress test...')
    sys.stdout.flush()
    
    try:
        data_dict = abm11.run_simulation()
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
        plt.savefig(os.path.join(out_dir, 'deformation_vs_time.png'))
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
        parallel=True,
        max_workers=24
    )
