import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import differential_evolution

R0 = 1.0

sol_exact = lambda t: R0 * np.exp(2*t)

t_max = 5.0
dt = 1e-3
t = np.arange(0, t_max, dt)

# Output directory
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join('data', 'diff_ev_test', f'run_{timestamp}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def numerical_solution(t, k):
    k = float(k)
    R = np.zeros(len(t))
    R[0] = R0
    for i in range(1, len(t)):
        R[i] = R[i-1] + k*R[i-1]*dt + np.random.normal(0, 0.01) # 0.005
    return R

def objective_function(k):
    # Accept k as scalar or 1-element array/list
    k = float(k)
    R = numerical_solution(t, k)
    return np.sqrt(((R - sol_exact(t)) ** 2).mean()) #np.mean((R - sol_exact(t))**2)

# Track convergence
error_history = []
k_history = []

def de_callback(xk, convergence):
    # xk is the best parameter vector at the current iteration
    current_k = float(xk[0]) if np.ndim(xk) else float(xk)
    k_history.append(current_k)
    error_history.append(float(objective_function(current_k)))
    return False  # continue optimization

result = differential_evolution(
    objective_function,
    [(0, 10)],
    maxiter=50,
    popsize=10,
    # seed=42,
    callback=de_callback,
    polish=True,
)

best_k = float(result.x[0])
best_error = float(result.fun)

print(f"Best k: {best_k:.6f}")
print(f"Best error: {best_error:.6f}")
print(f"Iterations (nit): {result.nit}")
print(f"Function evals (nfev): {result.nfev}")

# Plot 1: solution trajectories for the best parameter
fig1 = plt.figure()
plt.plot(t, numerical_solution(t, best_k), label='Numerical')
plt.plot(t, sol_exact(t), label='Exact', linestyle='--')
plt.title('Solution Comparison')
plt.xlabel('t')
plt.ylabel('R(t)')
plt.legend()
plt.grid(True, alpha=0.3)
fig1.savefig(os.path.join(OUTPUT_DIR, 'solution_comparison.png'), dpi=150, bbox_inches='tight')
# plt.show()

# Plot 2: convergence of k and error
fig2 = plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(k_history) + 1), k_history, marker='o', color='tab:red')
plt.axhline(y=2.0, color='black', linestyle='--', linewidth=2, alpha=0.8, label='True k')
plt.legend()
plt.title('k Convergence')
plt.xlabel('Iteration')
plt.ylabel('k')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(error_history) + 1), error_history, marker='o')
plt.title('Error Convergence')
plt.xlabel('Iteration')
plt.ylabel('Objective Error')
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'convergence.png'), dpi=150, bbox_inches='tight')
# plt.show()

# Animation: show numerical vs analytical as the best k evolves by iteration
if len(k_history) > 0:
    fig_anim, ax_anim = plt.subplots()
    ax_anim.plot(t, sol_exact(t), label='Exact')
    num_line, = ax_anim.plot([], [], label='Numerical')
    ax_anim.set_xlabel('t')
    ax_anim.set_ylabel('R(t)')
    ax_anim.set_title('Convergence Animation')
    ax_anim.grid(True, alpha=0.3)
    ax_anim.legend()

    # Set y-limits based on exact solution with a small margin
    y_exact = sol_exact(t)
    y_min = float(np.min(y_exact)) - 0.1 * abs(float(np.min(y_exact)))
    y_max = float(np.max(y_exact)) + 0.1 * abs(float(np.max(y_exact)))
    ax_anim.set_xlim(t[0], t[-1])
    ax_anim.set_ylim(y_min, y_max if y_max > y_min else y_min + 1.0)

    def init_anim():
        num_line.set_data([], [])
        return num_line,

    def update_anim(i):
        k_i = float(k_history[i])
        R_i = numerical_solution(t, k_i)
        num_line.set_data(t, R_i)
        err_i = float(error_history[i]) if i < len(error_history) else np.nan
        ax_anim.set_title(f'Iteration {i+1}/{len(k_history)}  |  k={k_i:.4f}  |  error={err_i:.4f}')
        return num_line,

    anim = FuncAnimation(
        fig_anim,
        update_anim,
        frames=len(k_history),
        init_func=init_anim,
        interval=300,
        blit=False
    )

    # Save animation to MP4 with GIF fallback
    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=10, bitrate=1800)
        anim.save(os.path.join(OUTPUT_DIR, 'convergence.mp4'), writer=writer)
    except Exception:
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=10)
            anim.save(os.path.join(OUTPUT_DIR, 'convergence.gif'), writer=writer)
        except Exception:
            pass

    # plt.show()