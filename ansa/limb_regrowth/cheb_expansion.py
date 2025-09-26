import numpy as np
import matplotlib.pyplot as plt
from scipy.special import chebyt
from scipy.integrate import quad, simpson
from numpy.polynomial.chebyshev import chebgauss
# from sklearn.preprocessing import MinMaxScaler


def omega(x):
    """Chebyshev weight function with singularity protection"""
    x = np.clip(x, -1 + 1e-15, 1 - 1e-15)  # Avoid exactly 1 or -1
    return 1 / np.sqrt(1 - x**2)

# cheb_basis = lambda n, x: chebyt(n, x)
def cheb_basis(n=0):
    return lambda x: chebyt(n)(x)

def chebyshev_inner_product(f, g, n_points=50):
    """
    Inner product using Gauss-Chebyshev quadrature.
    This is the correct way to integrate with weight omega(x).
    """
    nodes, weights = chebgauss(n_points)
    return np.sum(f(nodes) * g(nodes) * weights)

def inner_product(f, g, omega, type='data'):
    """
    Compute the inner product of two smooth functions f and g on the interval [-1, 1] with respect to the weight function omega.

    Args:
        f: numpy array of shape (n, 2)
        g: function
        omega: function
        type: 'data' or 'function'
            - 'data' if f is a numpy array of shape (n, 2), g is a function
            - 'function' if f is a function, g is a function

    Returns:
        float
    """
    if type == 'data':
        # For data, interpolate to Gauss-Chebyshev nodes for proper integration
        nodes, weights = chebgauss(len(f))
        f_interp = np.interp(nodes, f[:, 0], f[:, 1])
        return np.sum(f_interp * g(nodes) * weights)
    elif type == 'function':
        # Use proper Gauss-Chebyshev quadrature for functions
        return chebyshev_inner_product(f, g)

def scale_shape(data, target_range=(-1, 1)):
    """
    Scale data to target range while preserving shape/aspect ratio.
    Maps both x and y coordinates to target range using same scale factor.
    """
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    
    # Use the larger range to determine scale factor (preserves aspect ratio)
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    
    # Scale factor based on the larger dimension
    scale_factor = (target_range[1] - target_range[0]) / max_range
    
    # Center both dimensions and apply same scaling
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    scaled_data = data.copy()
    scaled_data[:, 0] = (data[:, 0] - x_center) * scale_factor
    scaled_data[:, 1] = (data[:, 1] - y_center) * scale_factor
    
    return scaled_data, scale_factor, x_center, y_center

def coefficients(f, n, type='data', return_scaling=False, normalize=False):
    """
    Compute the first n coefficients of the Chebyshev expansion of a function f.
    
    Args:
        f: data array or function
        n: number of coefficients
        type: 'data' or 'function'
        return_scaling: if True, also return scaling parameters for data type
        normalize: if True, normalize the coefficients the max coefficient to 1
    """
    # For data, ensure shape-preserving scaling
    if type == 'data':
        # Sort data by x-coordinate to ensure proper numerical integration
        f_sorted = f[np.argsort(f[:, 0])]
        f_scaled, scale_factor, x_offset, y_offset = scale_shape(f_sorted)
    else:
        f_scaled = f
        
    coefficients = []
    for i in range(n):
        Tn = cheb_basis(i)
        
        # Now both data and function types use Gauss-Chebyshev integration
        numerator = inner_product(f_scaled, Tn, omega, type=type)
        denominator = chebyshev_inner_product(Tn, Tn)  # Always use proper integration for denominator
        
        coefficients.append(numerator / denominator)

    if normalize:
        max_coeff = max(abs(c) for c in coefficients)
        coefficients = [c / max_coeff for c in coefficients]
    if type == 'data' and return_scaling:
        return coefficients, (scale_factor, x_offset, y_offset)
    else:
        return coefficients

def cheb_expansion(coefficients, n):
    """
    Compute the Chebyshev expansion of a function f.
    """
    return lambda x: sum(coefficients[i] * cheb_basis(i)(x) for i in range(n))

def inner_product_compare(coeffs1, coeffs2):
    """
    Compare the inner product of two sets of coefficients.
    """
    f1 = cheb_expansion(coeffs1, len(coeffs1))
    f2 = cheb_expansion(coeffs2, len(coeffs2))
    return inner_product(f1, f2, omega, type='function')
    # return np.sum(coeffs1 * coeffs2) # coefficient inner product

if __name__ == "__main__":
    # Test with T_2(x) = 2x^2 - 1 (should give c_2 = 1, all others = 0)
    # f1 = lambda x: 2*x**2 - 1
    # f = lambda x: np.log(1 + x)
    f = lambda x: np.sin(2*x)

    n = 10

    coeffs = coefficients(f, n, type='function')
    print(f"Coefficients for f(x)")
    for i, c in enumerate(coeffs):
        if abs(c) > 1e-10:
            print(f"c_{i} = {c:.6f}")
        else:
            print(f"c_{i} ≈ 0")

    # # Test with another function: cos(πx/2)
    # print("\nCoefficients for cos(πx/2):")
    # f2 = lambda x: np.cos(np.pi * x / 2)
    # coeffs2 = coefficients(f2, n)
    # for i, c in enumerate(coeffs2):
    #     if abs(c) > 1e-6:
    #         print(f"c_{i} = {c:.6f}")

    x = np.linspace(-1+1e-10, 1-1e-10, 100)
    plt.plot(x, f(x), label='f(x) = sin(2x)')

    plt.plot(x, cheb_expansion(coefficients(f, n, type='function'), n)(x), 'r--', label=f'Chebyshev expansion (n={n})')
    plt.legend()
    plt.show()

    plt.bar(range(n), coeffs, color='b')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xticks(range(n))
    max_coeff = max(abs(np.array(coeffs)))
    y_min = -np.ceil(max_coeff * 10) / 10 - 0.2
    y_max = np.ceil(max_coeff * 10) / 10 + 0.2
    plt.yticks(np.arange(y_min, y_max + 0.05, 0.2))
    plt.ylim(y_min, y_max)
    plt.xlabel('n')
    plt.ylabel('Coefficient')
    plt.title('Chebyshev Coefficients')
    plt.show()