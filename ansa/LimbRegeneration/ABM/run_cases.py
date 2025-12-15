"""
Run all simulation cases sequentially or in parallel.
"""
import multiprocessing as mp
import importlib
import os
import sys
import numpy as np
from datetime import date
import warnings
import pickle

warnings.filterwarnings('ignore')

def run_single_case(case_name, soft, parent_dir):
    """Run a single simulation case."""
    import config
    import abm11
    
    config.ALLOW_SOFTENING = soft
    config.MIGRATION_ENABLED = False
    config.MIGRATION_PERCENT = 0.0
    config.INTERCALATION_ENABLED = False
    config.INTERCAL_PERCENT = 0.0
    config.GRADIENT = None
    config.DIRECTED_DIVISION_ANGLE = None
    config.JAMMING_ENABLED = False
    config.EXT_STRESS_FORCE = False
    config.VIDEO_FLAG = True
    
    folder = 'SOFT' if soft else 'NO_SOFT'
    config.OUTPUT_DIR = os.path.join(parent_dir, folder, case_name)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    if case_name == 'ANGLE0':
        config.DIRECTED_DIVISION_ANGLE = 0.0
    elif case_name == 'ANGLE_PI_2':
        config.DIRECTED_DIVISION_ANGLE = np.pi/2
    elif case_name == 'MIGRATION6':
        config.MIGRATION_ENABLED = True
        config.MIGRATION_PERCENT = 6.0
    elif case_name == 'MIGRATION18':
        config.MIGRATION_ENABLED = True
        config.MIGRATION_PERCENT = 18.0
    elif case_name == 'MIGRATION25':
        config.MIGRATION_ENABLED = True
        config.MIGRATION_PERCENT = 25.0
    elif case_name == 'MIGRATION30':
        config.MIGRATION_ENABLED = True
        config.MIGRATION_PERCENT = 30.0
    elif case_name == 'MIGRATION35':
        config.MIGRATION_ENABLED = True
        config.MIGRATION_PERCENT = 35.0
    elif case_name == 'MIGRATION50':
        config.MIGRATION_ENABLED = True
        config.MIGRATION_PERCENT = 50.0
    elif case_name == 'JAMMING0':
        config.JAMMING_ENABLED = True
    elif case_name == 'JAMMING6':
        config.JAMMING_ENABLED = True
        config.MIGRATION_ENABLED = True
        config.MIGRATION_PERCENT = 6.0
    elif case_name == 'JAMMING18':
        config.JAMMING_ENABLED = True
        config.MIGRATION_ENABLED = True
        config.MIGRATION_PERCENT = 18.0
    elif case_name == 'INTERCAL6':
        config.INTERCALATION_ENABLED = True
        config.INTERCAL_PERCENT = 6.0
    elif case_name == 'INTERCAL18':
        config.INTERCALATION_ENABLED = True
        config.INTERCAL_PERCENT = 18.0
    elif case_name == 'LINEAR':
        config.GRADIENT = 'linear'
    elif case_name == 'ZONE':
        config.GRADIENT = 'zone'
    
    print(f"[{folder}/{case_name}] Starting simulation...")
    sys.stdout.flush()
    
    try:
        importlib.reload(abm11)
        abm11.run_simulation()
        print(f"[{folder}/{case_name}] Complete!")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"[{folder}/{case_name}] ERROR: {e}")
        sys.stdout.flush()
        return False

def run_all_cases(parallel=False, max_workers=None):
    """
    Run all cases sequentially or in parallel.
    
    Args:
        parallel: If True, run in parallel (default: False)
        max_workers: Max parallel workers (None = all CPU cores)
    """
    today = date.today()
    parent_dir = os.path.join('data', 'output', f'Cases{today}')
    os.makedirs(parent_dir, exist_ok=True)
    
    cases = [
        'DEFAULT', 'ANGLE0', 'ANGLE_PI_2', 
        'MIGRATION6', 'MIGRATION18', 'MIGRATION25', 'MIGRATION35', 'MIGRATION50',
        'JAMMING0', 'JAMMING6', 'JAMMING18', 
        'INTERCAL6', 'INTERCAL18',
        'LINEAR', 'ZONE'
    ]
    
    jobs = []
    for case in cases:
        jobs.append((case, True, parent_dir))
        jobs.append((case, False, parent_dir))
    
    total_jobs = len(jobs)
    print(f"{'='*60}")
    print(f"Running {total_jobs} cases {'in parallel' if parallel else 'sequentially'}")
    print(f"Output directory: {parent_dir}")
    if parallel:
        if max_workers is None:
            print(f"Using all available CPU cores: {mp.cpu_count()}")
        else:
            print(f"Using {max_workers} parallel workers")
    print(f"{'='*60}\n")
    
    if parallel:
        with mp.Pool(processes=max_workers) as pool:
            results = pool.starmap(run_single_case, jobs)
    else:
        results = [run_single_case(*job) for job in jobs]
    
    print(f"\n{'='*60}")
    print("All cases complete!")
    print(f"Successful: {sum(results)}/{total_jobs}")
    print(f"Failed: {total_jobs - sum(results)}/{total_jobs}")
    print(f"{'='*60}")
    
    print("\nGenerating plots...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        from utils.make_bar_plots import create_bar_plot#, extract_morphometrics_from_metadata
        from utils.plot_all_cell_counts import plot_all_cell_counts
        
        print("Creating bar plots...")
        length_dict_soft = {}
        length_dict_no_soft = {}
        area_dict_soft = {}
        area_dict_no_soft = {}
        
        for case in cases:
            for soft_type in ['SOFT', 'NO_SOFT']:
                case_dir = os.path.join(parent_dir, soft_type, case)
                # metadata_path = os.path.join(case_dir, 'metadata.txt')
                # if os.path.exists(metadata_path):

                with open(f'{case_dir}/data_dict.pkl', 'rb') as f:
                    data_dict = pickle.load(f)
                    area, perimeter, AR_whole_limb, AR_outgrowth, ellipticity, roundness, length, b, volume_fraction = data_dict['morphometrics_final']

                    if soft_type == 'SOFT':
                        if length is not None:
                            length_dict_soft[case] = length
                        if area is not None:
                            area_dict_soft[case] = area
                    else:
                        if length is not None:
                            length_dict_no_soft[case] = length
                        if area is not None:
                            area_dict_no_soft[case] = area
        
        create_bar_plot(length_dict_soft, length_dict_no_soft, 'Length', output_dir=parent_dir)
        create_bar_plot(area_dict_soft, area_dict_no_soft, 'Area', output_dir=parent_dir)
        print("Bar plots complete!")
        
        print("Creating cell count comparison plot...")
        plot_all_cell_counts(parent_dir)
        print("Cell count plot complete!")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    return results

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    results = run_all_cases(parallel=True, max_workers=16)
    

