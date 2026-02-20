import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np


data = loadmat('Xb_CTRL_T2_final.mat')
Xb_CTRL_T2 = data['Xb_CTRL_T2']

x = Xb_CTRL_T2[:,0]
y = Xb_CTRL_T2[:,1]

coefficients = np.polyfit(x, y, 5)
polynomial = np.poly1d(coefficients)

x_fit = np.linspace(x.min(), x.max(), 500)
y_fit = polynomial(x_fit)

# plt.plot(x,y)

plt.scatter(x, y, marker = 'x')
plt.plot(x_fit, y_fit, color = 'red')
plt.show()

