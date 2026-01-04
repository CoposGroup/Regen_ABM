"""Profiling utilities for performance analysis"""
import cProfile
import pstats
from pstats import SortKey
import os

def profiling(run_simulation_func, output_dir, print_stats=True):
    """Run simulation with profiling enabled"""    
    profiler = cProfile.Profile()
    profiler.enable()
    data_dict = run_simulation_func()
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    
    if print_stats:
        print("\n--- cProfile Results (Top 20 by cumulative time) ---")
        stats.print_stats(20)
        
        print("\n--- cProfile Results (Top 20 by total time) ---")
        stats.sort_stats(SortKey.TIME)
        stats.print_stats(20)
    
    profile_file = os.path.join(output_dir, 'cell_sim_profile.prof')
    stats.dump_stats(profile_file)
    print(f"Detailed profile saved to {profile_file}")
    
    return data_dict