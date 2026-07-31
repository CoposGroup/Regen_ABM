import numpy as np

blp0 = np.load("src/input/sim_data/blp0.npy")
blm0 = np.load("src/input/sim_data/blm0.npy")

# print(blp0)
# print()
# print(blm0)

print(blm0 == np.roll(blp0, 1))