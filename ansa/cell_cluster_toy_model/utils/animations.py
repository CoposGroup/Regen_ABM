"""
Animation utilities for limb regeneration simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
from config import DL_CRIT
Xb = np.loadtxt('data/input/bone.csv', delimiter=',')
Xe0 = np.loadtxt('data/input/epi0.csv', delimiter=',')
x_cut = Xe0[:, 0].max()

class AnimationManager:
    def __init__(self, output_dir, video_params, video_flag=True):
        self.output_dir = output_dir
        self.video_flag = video_flag
        
        if video_flag:
            # Main animation setup
            self.fig, self.ax = plt.subplots(
                figsize=video_params['figsize'], 
                dpi=video_params['dpi']
            )
            self.fig.canvas.draw()
            w, h = self.fig.canvas.get_width_height()
            
            # Density animation setup
            self.fig_density, self.ax_density = plt.subplots(
                figsize=(12, 8), 
                dpi=video_params['dpi']
            )
            self.fig_density.canvas.draw()
            w_density, h_density = self.fig_density.canvas.get_width_height()
            
            # Video writers
            self.writer = imageio.get_writer(
                f'{output_dir}/out.mp4',
                fps=video_params['fps'],
                ffmpeg_params=['-s', f'{w}x{h}']
            )
            
            self.writer_density = imageio.get_writer(
                f'{output_dir}/density_animation.mp4',
                fps=video_params['fps'],
                ffmpeg_params=['-s', f'{w_density}x{h_density}']
            )
        else:
            self.writer = None
            self.writer_density = None
    
    def animate_frame(self, step, t, pos, Xe, Xb, pos0, division_status, soft_idx, frame_skip, x_bounds=(-1, 3), y_bounds=(-2, 2), scale=np.pi*DL_CRIT**2, boundary=True, forces=None, migrant_cells=None, intercal_cells=None):
        """Draw current frame and record to video if enabled"""
        # if not self.video_flag or (step % frame_skip != 0) or step == 0:
        #     return

        active = ~np.isnan(pos[:, 0])
        daughters = division_status

        self.ax.clear()
        
        # Plot boundary
        if boundary:
            for i in range(len(Xe)-1):
                if i in soft_idx:
                    color = 'pink'  # soft boundary segments
                else:
                    color = 'red'   # hard boundary segments
                    
                self.ax.plot([Xe[i,0], Xe[i+1,0]], 
                            [Xe[i,1], Xe[i+1,1]], 
                            '-', lw=2, color=color)
            
            # self.ax.plot(Xb, color='black')
            for i in range(len(Xb)-1):
                self.ax.plot([Xb[i,0], Xb[i+1,0]], 
                            [Xb[i,1], Xb[i+1,1]], 
                            '-', lw=3, color='black', alpha=0.8)
            # Close the bone polygon
            self.ax.plot([Xb[-1,0], Xb[0,0]], 
                        [Xb[-1,1], Xb[0,1]], 
                        '-', lw=3, color='black', alpha=0.8)
            self.ax.plot([x_cut, x_cut], [-1.25, 1.25], '--', lw=1, color='k')



        

        cell_radius = DL_CRIT / 2  # in data units

        # Convert radius from data units to points
        # Get the scaling factor from data coordinates to display coordinates
        bbox = self.ax.get_window_extent().transformed(self.ax.figure.dpi_scale_trans.inverted())
        width_inch = bbox.width
        data_width = x_bounds[1] - x_bounds[0]
        points_per_data_unit = (width_inch * 72) / data_width  # 72 points per inch

        # Calculate radius in points, then area in points^2
        r_points = cell_radius * points_per_data_unit  # Fixed: removed extra /2
        area_points2 = np.pi * r_points**2

        self.ax.scatter(pos[active & ~daughters, 0], pos[active & ~daughters, 1], s=area_points2,
                    facecolor=(170/255, 157/255, 241/255), edgecolors='black')  # Original cells
        self.ax.scatter(pos[active & daughters, 0], pos[active & daughters, 1], s=area_points2,
                    facecolor=(128/255, 0/255, 128/255), edgecolors='black')  # Daughter cells (purple)
        if migrant_cells is not None:
            self.ax.scatter(pos[active & migrant_cells, 0], pos[active & migrant_cells, 1], s=area_points2,
                        facecolor='red')#, edgecolors='black')  # Migrant Cells (red))
        # if intercal_cells is not None:
        #     self.ax.scatter(pos[active & intercal_cells, 0], pos[active & intercal_cells, 1], s=area_points2,
        #                 facecolor='green')#, edgecolors='black')  # Intercal Cells (green))
        # plot forces
        if forces is not None:
            for i in range(len(pos)):
                if active[i]:
                    force = forces[i]
                    self.ax.quiver(
                        pos[i, 0], pos[i, 1],
                        force[0], force[1],
                        angles='xy', scale_units='xy',
                        width=0.01, color='k', alpha=0.5, scale=10, headwidth=0.5, headlength=0.5
            )

            
        # Amputation plane
        
        # Scale bar
        sbx = np.linspace(0.7, 0.7 + x_cut/5, 100)
        self.ax.plot(sbx, -1.5 * np.ones_like(sbx), '-', lw=5, color='w')

        self.ax.set_xlim(x_bounds)
        self.ax.set_ylim(y_bounds)
        self.ax.grid(True)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_aspect('equal', 'box')
        self.ax.set_title(f"T = {t:.4f}")


        # Capture frame
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.renderer.buffer_rgba())[..., :3]
        self.writer.append_data(buf)
    
    def animate_density_heatmap(self, step, t, pos, Xe, Xb, soft_range, x_cut, bin_size=0.1, frame_skip=10, boundary=True, x_bounds=(-1, 3), y_bounds=(-2, 2)):
        """Create animated density heatmap frames"""
        if not self.video_flag or not self.writer_density or (step % frame_skip != 0) or step == 0:
            return
        
        # Get active cells only
        active = ~np.isnan(pos[:, 0])
        pos_active = pos[active]
        
        if len(pos_active) == 0:
            return
        
        # Create edges for histogram
        x_edges = np.arange(x_bounds[0], x_bounds[1] + bin_size, bin_size)
        y_edges = np.arange(y_bounds[0], y_bounds[1] + bin_size, bin_size)
        
        # Calculate density
        H, xedges, yedges = np.histogram2d(
            pos_active[:, 0], pos_active[:, 1],
            bins=[x_edges, y_edges]
        )
        
        # Create centered coordinates for pcolor
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
        
        # Clear previous plot
        self.ax_density.clear()
        
        # Plot heatmap
        pcm = self.ax_density.pcolormesh(X, Y, H, cmap='plasma', shading='auto', vmin=0, vmax=20)
        
        # Add colorbar only once (check if it exists)
        if not hasattr(self, 'colorbar_added'):
            self.fig_density.colorbar(pcm, ax=self.ax_density, label='Cell count per bin')
            self.colorbar_added = True
        
        # Add amputation plane
        
        # Plot boundary segments with appropriate colors
        if boundary:
            self.ax_density.axvline(x=x_cut, color='black', linestyle='--', alpha=0.8, linewidth=2)

            for idx in range(len(Xe)-1):
                if Xe[idx, 1] > soft_range[0] and Xe[idx, 1] < soft_range[1]:
                    color = 'pink'
                else:
                    color = 'red'
                
                self.ax_density.plot([Xe[idx,0], Xe[idx+1,0]], 
                            [Xe[idx,1], Xe[idx+1,1]], 
                            '-', lw=2, color=color, alpha=0.8)
                
            for i in range(len(Xb)-1):
                self.ax.plot([Xb[i,0], Xb[i+1,0]], 
                            [Xb[i,1], Xb[i+1,1]], 
                            '-', lw=3, color='black', alpha=0.8)
            # Close the bone polygon
            self.ax.plot([Xb[-1,0], Xb[0,0]], 
                        [Xb[-1,1], Xb[0,1]], 
                        '-', lw=3, color='black', alpha=0.8)
        # Customize plot
        self.ax_density.set_xlabel('x')
        self.ax_density.set_ylabel('y')
        self.ax_density.set_title(f'Cell Density Distribution - T = {t:.3f}')
        self.ax.set_xlim(x_bounds)
        self.ax.set_ylim(y_bounds)
        self.ax_density.set_aspect('equal')
        self.ax_density.grid(True, alpha=0.3)
        
        # Add statistics text box
        active_cells = len(pos_active)
        max_density = H.max()
        mean_density = H[H>0].mean() if np.any(H>0) else 0
        
        stats_text = (f'Total cells: {active_cells}\n'
                     f'Max density: {max_density:.0f}\n'
                     f'Mean density: {mean_density:.1f}\n'
                     f'Time: {t:.3f}')
        
        self.ax_density.text(0.02, 0.98, stats_text,
                           transform=self.ax_density.transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Draw and capture frame
        self.fig_density.canvas.draw()
        buf = np.asarray(self.fig_density.canvas.renderer.buffer_rgba())[..., :3]
        self.writer_density.append_data(buf)
    
    def close(self):
        """Clean up animation resources"""
        if self.video_flag:
            if self.writer:
                self.writer.close()
            if self.writer_density:
                self.writer_density.close()
            plt.close(self.fig)
            plt.close(self.fig_density)