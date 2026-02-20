"""Run all simulation cases sequentially or in parallel."""
import _setup_path
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
    import config, abm
    
    config.SPORATIC_SOFTENING = False
    config.ALLOW_SOFTENING = soft
    config.MIGRATION_ENABLED = False
    config.MIGRATION_FRACTION = 0.0
    config.INTERCALATION_ENABLED = False
    config.INTERCAL_FRACTION = 0.0
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
        config.MIGRATION_FRACTION = 0.06
    elif case_name == 'MIGRATION18':
        config.MIGRATION_ENABLED = True
        config.MIGRATION_FRACTION = 0.18
    elif case_name == 'MIGRATION25':
        config.MIGRATION_ENABLED = True
        config.MIGRATION_FRACTION = 0.25
    elif case_name == 'MIGRATION30':
        config.MIGRATION_ENABLED = True
        config.MIGRATION_FRACTION = 0.3
    elif case_name == 'MIGRATION35':
        config.MIGRATION_ENABLED = True
        config.MIGRATION_FRACTION = 0.35
    elif case_name == 'MIGRATION50':
        config.MIGRATION_ENABLED = True
        config.MIGRATION_FRACTION = 0.5
    elif case_name == 'JAMMING0':
        config.JAMMING_ENABLED = True
    elif case_name == 'JAMMING6':
        config.JAMMING_ENABLED = True
        config.MIGRATION_ENABLED = True
        config.MIGRATION_FRACTION = 0.6
    elif case_name == 'JAMMING18':
        config.JAMMING_ENABLED = True
        config.MIGRATION_ENABLED = True
        config.MIGRATION_FRACTION = 0.18
    elif case_name == 'JAMMING25':
        config.JAMMING_ENABLED = True
        config.MIGRATION_ENABLED = True
        config.MIGRATION_FRACTION = 0.25
    elif case_name == 'JAMMING30':
        config.JAMMING_ENABLED = True
        config.MIGRATION_ENABLED = True
        config.MIGRATION_FRACTION = 0.3
    elif case_name == 'JAMMING50':
        config.JAMMING_ENABLED = True
        config.MIGRATION_ENABLED = True
        config.MIGRATION_FRACTION = 0.5
    elif case_name == 'INTERCAL6':
        config.INTERCALATION_ENABLED = True
        config.INTERCAL_FRACTION = 0.06
    elif case_name == 'INTERCAL18':
        config.INTERCALATION_ENABLED = True
        config.INTERCAL_FRACTION = 0.18
    elif case_name == 'INTERCAL30':
        config.INTERCALATION_ENABLED = True
        config.INTERCAL_FRACTION = 0.3
    elif case_name == 'GRADIENT':
        config.GRADIENT = True
    
    print(f"[{folder}/{case_name}] Starting simulation...")
    sys.stdout.flush()
    
    try:
        importlib.reload(abm)
        abm.run_simulation()
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
    parent_dir = f'Cases{today}'
    os.makedirs(parent_dir, exist_ok=True)
    

    # cases = [
    #     'DEFAULT', 'ANGLE0', 'ANGLE_PI_2', 
    #     'MIGRATION6', 'MIGRATION18', 'MIGRATION25', 'MIGRATION30', 'MIGRATION35', 'MIGRATION50',
    #     'JAMMING0', 'JAMMING6', 'JAMMING18', 'JAMMING25', 'JAMMING30',
    #     'INTERCAL6', 'INTERCAL18', 'INTERCAL30',
    #     'GRADIENT'
    # ]

    cases = ['DEFAULT', 'ANGLE0', 'ANGLE_PI_2', 'MIGRATION18', 'MIGRATION30', 'MIGRATION50',
            'JAMMING0', 'JAMMING18', 'JAMMING30', 'INTERCAL18', 'INTERCAL30', 'GRADIENT']

    
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
        from utils.post_process import cases_bar_plot
        from utils.post_process import plot_all_cell_counts
        
        print("Creating bar plots...")
        length_dict_soft = {}
        length_dict_no_soft = {}
        area_dict_soft = {}
        area_dict_no_soft = {}
        AR_dict_soft = {}
        AR_dict_no_soft = {}        
        
        for case in cases:
            for soft_type in ['SOFT', 'NO_SOFT']:
                case_dir = os.path.join(parent_dir, soft_type, case)

                with open(f'{case_dir}/data_dict.pkl', 'rb') as f:
                    data_dict = pickle.load(f)
                    area, perimeter, AR_whole_limb, AR_outgrowth, ellipticity, roundness, length, b, volume_fraction = data_dict['morphometrics_final']

                    if soft_type == 'SOFT':
                        if length is not None:
                            length_dict_soft[case] = length
                        if area is not None:
                            area_dict_soft[case] = area
                        if AR_outgrowth is not None:
                            AR_dict_soft[case] = AR_outgrowth
                    else:
                        if length is not None:
                            length_dict_no_soft[case] = length
                        if area is not None:
                            area_dict_no_soft[case] = area
                        if AR_outgrowth is not None:
                            AR_dict_no_soft[case] = AR_outgrowth
        
        cases_bar_plot(length_dict_soft, length_dict_no_soft, 'Length', output_dir=parent_dir)
        cases_bar_plot(area_dict_soft, area_dict_no_soft, 'Area', output_dir=parent_dir)
        cases_bar_plot(AR_dict_soft, AR_dict_no_soft, 'Aspect Ratio', output_dir=parent_dir)
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
    results = run_all_cases(parallel=True, max_workers=30)