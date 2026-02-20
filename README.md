# Agent Based Model For Axolotl Limb Regeneration

The important files are:
- requirements.txt - Python package dependencies

src/
- abm.py - Main simulation code
- config.py - Default parameter configuration
src/utils
- Contains plotting and data post-processing functions

scripts/
- run_sim.py - Runs a simulation with chosen parameters
- run_cases.py - Runs the cases from figure 3A, 3B, 3C, S2A, S2B
- stress_test.py - Runs poking experiments as seen in figure 5
- param_optimization_pso.py - Runs particle swarm optimization (PSO) pipeline, various parameters can be set
- read_pso_optim.py - Allows the user to check the status of a PSO run with simple visualizations
- pso_figs.py - Generates the scatter plots seen in figure 7