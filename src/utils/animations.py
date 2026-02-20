"""Animation utilities for limb regeneration simulation"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio
from config import D0, BONE_VISUALIZATION, XMIN, XMAX, YMIN, YMAX, CONVERSION_FACTOR_UM
from matplotlib.colors import ListedColormap
from matplotlib import cm

def truncate_colormap(cmap, minval=0.2, maxval=1.0, n=256):
    new_cmap = ListedColormap(cmap(np.linspace(minval, maxval, n)))
    return new_cmap

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
            video_path = os.path.join(output_dir, 'sim.mp4')
            density_video_path = os.path.join(output_dir, 'density_animation.mp4')
            
            self.writer = imageio.get_writer(
                video_path,
                fps=video_params['fps'],
                ffmpeg_params=['-s', f'{w}x{h}']
            )
            
            self.writer_density = imageio.get_writer(
                density_video_path,
                fps=video_params['fps'],
                ffmpeg_params=['-s', f'{w_density}x{h_density}']
            )
        else:
            self.writer = None
            self.writer_density = None

    def animate_frame(self, step, t, pos, Xe, Xb, cycle_phases, kb_vals=None, x_bounds=(XMIN, XMAX), y_bounds=(YMIN, YMAX), scale=np.pi*D0**2, boundary=True, forces=None, migrant_cells=None, intercal_cells=None, jammed_cells=None, x_cut=0, show_real_units=True, regulation_front=None, title_suffix=''):
        """Draw current frame and record to video"""
        if not self.video_flag or not self.writer:
            return

        active_mask = ~np.isnan(pos[:, 0])
        phase0 = np.where((cycle_phases == 0) & active_mask)[0]
        phase1 = np.where((cycle_phases == 1) & active_mask)[0]
        self.ax.clear()
        
        if show_real_units:
            x_bounds_display = (x_bounds[0] * CONVERSION_FACTOR_UM, x_bounds[1] * CONVERSION_FACTOR_UM)
            y_bounds_display = (y_bounds[0] * CONVERSION_FACTOR_UM, y_bounds[1] * CONVERSION_FACTOR_UM)
            pos_display = pos * CONVERSION_FACTOR_UM
            Xe_display = Xe * CONVERSION_FACTOR_UM
            if BONE_VISUALIZATION:
                Xb_display = Xb * CONVERSION_FACTOR_UM
        else:
            x_bounds_display = x_bounds
            y_bounds_display = y_bounds
            pos_display = pos
            Xe_display = Xe
            if BONE_VISUALIZATION:
                Xb_display = Xb
        
        # Plot boundary
        if boundary:
            if kb_vals is not None:
                from config import KAPPA0, KAPPA2
                vmin = 1.0
                vmax = 150.0
                
                # Use normalized color mapping for all cases
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                cmap = truncate_colormap(cm.get_cmap('PuRd'), 0.2, 1.0)
                for i in range(len(Xe)-1):
                    color = cmap(norm(kb_vals[i]))
                    self.ax.plot([Xe_display[i,0], Xe_display[i+1,0]], 
                                [Xe_display[i,1], Xe_display[i+1,1]], 
                                '-', lw=2, color=color)

            if BONE_VISUALIZATION:
                for i in range(len(Xb)-1):
                    self.ax.plot([Xb_display[i,0], Xb_display[i+1,0]], 
                                [Xb_display[i,1], Xb_display[i+1,1]], 
                                '-', lw=3, color='black', alpha=0.8)
                self.ax.plot([Xb_display[-1,0], Xb_display[0,0]], 
                            [Xb_display[-1,1], Xb_display[0,1]], 
                            '-', lw=3, color='black', alpha=0.8)
            
            x_cut_display = x_cut * CONVERSION_FACTOR_UM if show_real_units else x_cut
            self.ax.axvline(x=x_cut_display, color='black', linestyle='--', alpha=0.8, linewidth=1, label='Amputation Plane')

        # Plot regulation front if provided (used for migration and/or proliferation gradient)
        from config import MIGRATION_ENABLED, MIGRATION_DELAY, REGULATION_FRONT_FLAG, GRADIENT
        if ((MIGRATION_ENABLED or GRADIENT == 'zone') and 
            REGULATION_FRONT_FLAG and
            regulation_front is not None and 
            not np.isinf(regulation_front) and 
            t >= MIGRATION_DELAY):
            migration_front_display = regulation_front * CONVERSION_FACTOR_UM if show_real_units else regulation_front
            self.ax.axvline(x=migration_front_display, color='purple', linestyle='--', alpha=0.8, linewidth=2, label='Regulation Front')
        
        cell_radius = D0 / 2
        
        # Cache cell size calculation after first frame to prevent size changes
        if not hasattr(self, 'cached_area_points2'):
            bbox = self.ax.get_window_extent().transformed(self.ax.figure.dpi_scale_trans.inverted())
            width_inch = bbox.width
            data_width = x_bounds[1] - x_bounds[0]
            points_per_data_unit = (width_inch * 72) / data_width  # 72 points per inch
            r_points = cell_radius * points_per_data_unit
            self.cached_area_points2 = np.pi * r_points**2
        
        area_points2 = self.cached_area_points2

        # Plot cells
        self.ax.scatter(pos_display[phase0, 0], pos_display[phase0, 1], s=area_points2, facecolor=(223/255, 224/255, 95/255), edgecolors='black')
        self.ax.scatter(pos_display[phase1, 0], pos_display[phase1, 1], s=area_points2, facecolor=(120/255, 237/255, 240/255), edgecolors='black')

        if REGULATION_FRONT_FLAG:
            currently_migrating = active_mask & migrant_cells & (pos[:, 0] > regulation_front)
        elif not REGULATION_FRONT_FLAG:
            currently_migrating = active_mask & migrant_cells
        if migrant_cells is not None:
            self.ax.scatter(pos_display[currently_migrating, 0], pos_display[currently_migrating, 1], s=area_points2, facecolor='none', edgecolors='purple', linewidths=1.5, label='Migration')
        if intercal_cells is not None:
            intercal_and_active = active_mask & intercal_cells
            self.ax.scatter(pos_display[intercal_and_active, 0], pos_display[intercal_and_active, 1], s=area_points2, facecolor='none', edgecolors='red', linewidths=1.5, label='Intercalation')
        if jammed_cells is not None:
            jammed_and_active = active_mask & jammed_cells
            self.ax.scatter(pos_display[jammed_and_active, 0], pos_display[jammed_and_active, 1], s=area_points2, facecolor='none', edgecolors='blue', linewidths=1.5, label='Jammed')

        # Plot forces
        if forces is not None:
            for i in range(len(pos)):
                if active_mask[i]:
                    force = forces[i]
                    force_scale = CONVERSION_FACTOR_UM if show_real_units else 1.0
                    self.ax.quiver(
                        pos_display[i, 0], pos_display[i, 1],
                        force[0] * force_scale, force[1] * force_scale,
                        angles='xy', scale_units='xy',
                        width=0.01, color='k', alpha=0.5, scale=10, headwidth=0.5, headlength=0.5
            )

        # Set final axis properties
        self.ax.set_xlim(x_bounds_display)
        self.ax.set_ylim(y_bounds_display)
        
        if show_real_units:
            self.ax.set_xlabel('x (um)', fontsize=12, fontweight='bold')
            self.ax.set_ylabel('y (um)', fontsize=12, fontweight='bold')
        else:
            self.ax.set_xlabel('x', fontsize=12)
            self.ax.set_ylabel('y', fontsize=12)
        
        self.ax.grid(False)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_title(f"{t:.4f} dpa")
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.renderer.buffer_rgba())[..., :3]
        self.writer.append_data(buf)
    
    def pause_animation(self, duration_seconds=1.0):
        """Pause the animation by writing duplicate frames."""
        if not self.video_flag or not self.writer:
            return
        
        buf = np.asarray(self.fig.canvas.renderer.buffer_rgba())[..., :3]
        from config import FPS
        num_frames = int(FPS * duration_seconds)
        
        # Write the same frame multiple times to create a pause effect
        for _ in range(num_frames):
            self.writer.append_data(buf)

    def animate_density_heatmap(self, step, t, pos, Xe, Xb, kb_vals, x_cut, bin_size=0.1, frame_skip=10, boundary=True, x_bounds=(XMIN, XMAX), y_bounds=(YMIN, YMAX), show_real_units=True):
        """Create animated density heatmap frames"""
        if not self.video_flag or not self.writer_density or (step % frame_skip != 0) or step == 0:
            return
        
        active = ~np.isnan(pos[:, 0])
        pos_active = pos[active]
        
        if len(pos_active) == 0:
            return
        
        if show_real_units:
            pos_active_display = pos_active * CONVERSION_FACTOR_UM
            x_bounds_display = (x_bounds[0] * CONVERSION_FACTOR_UM, x_bounds[1] * CONVERSION_FACTOR_UM)
            y_bounds_display = (y_bounds[0] * CONVERSION_FACTOR_UM, y_bounds[1] * CONVERSION_FACTOR_UM)
            bin_size_display = bin_size * CONVERSION_FACTOR_UM
            Xe_display = Xe * CONVERSION_FACTOR_UM
            if BONE_VISUALIZATION:
                Xb_display = Xb * CONVERSION_FACTOR_UM
            x_cut_display = x_cut * CONVERSION_FACTOR_UM
        else:
            pos_active_display = pos_active
            x_bounds_display = x_bounds
            y_bounds_display = y_bounds
            bin_size_display = bin_size
            Xe_display = Xe
            if BONE_VISUALIZATION:
                Xb_display = Xb
            x_cut_display = x_cut
        
        # Create edges for histogram and calculate density
        x_edges = np.arange(x_bounds_display[0], x_bounds_display[1] + bin_size_display, bin_size_display)
        y_edges = np.arange(y_bounds_display[0], y_bounds_display[1] + bin_size_display, bin_size_display)
        
        H, xedges, yedges = np.histogram2d(
            pos_active_display[:, 0], pos_active_display[:, 1],
            bins=[x_edges, y_edges]
        )
        
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
        
        self.ax_density.clear()
        bin_area_um2 = bin_size_display * bin_size_display
        H_real_units = H / bin_area_um2 * 1000  # Convert to cells per 1000 um^2
        
        vmax_sim_units = 8.0
        vmax_real_units = (vmax_sim_units / bin_area_um2) * 1000  # Convert to cells per 1000 um^2
        pcm = self.ax_density.pcolormesh(X, Y, H_real_units, cmap='plasma', shading='auto', vmin=0, vmax=vmax_real_units)
        
        if not hasattr(self, 'colorbar_added'):
            self.fig_density.colorbar(pcm, ax=self.ax_density, label='Cell density (cells/1000 um^2)')
            self.colorbar_added = True
            
        # Plot boundary
        if boundary:
            self.ax_density.axvline(x=x_cut_display, color='green', linestyle='--', alpha=0.8, linewidth=2)

            if kb_vals is not None:
                vmin = 1
                vmax = 150
                
                # Use normalized color mapping for all cases
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                cmap = truncate_colormap(cm.get_cmap('PuRd'), 0.2, 1.0)
                for i in range(len(Xe)-1):
                    color = cmap(norm(kb_vals[i]))
                    self.ax_density.plot([Xe_display[i,0], Xe_display[i+1,0]], 
                                       [Xe_display[i,1], Xe_display[i+1,1]], 
                                       '-', lw=2, color=color)
            if BONE_VISUALIZATION:
                for i in range(len(Xb)-1):
                    self.ax_density.plot([Xb_display[i,0], Xb_display[i+1,0]], 
                                        [Xb_display[i,1], Xb_display[i+1,1]], 
                                        '-', lw=3, color='black', alpha=0.8)
                # Close the bone polygon
                self.ax_density.plot([Xb_display[-1,0], Xb_display[0,0]], 
                                    [Xb_display[-1,1], Xb_display[0,1]], 
                                    '-', lw=3, color='black', alpha=0.8)
        # Customize plot
        if show_real_units:
            self.ax_density.set_xlabel('x (um)', fontsize=12, fontweight='bold')
            self.ax_density.set_ylabel('y (um)', fontsize=12, fontweight='bold')
        else:
            self.ax_density.set_xlabel('x', fontsize=12)
            self.ax_density.set_ylabel('y', fontsize=12)
        
        self.ax_density.set_title(f'{t:.3f} dpa')
        self.ax_density.set_xlim(x_bounds_display)
        self.ax_density.set_ylim(y_bounds_display)
        self.ax_density.set_aspect('equal')
        self.ax_density.grid(True, alpha=0.3)
        
        # Add statistics text box
        active_cells = len(pos_active)
        max_density = H_real_units.max()
        mean_density = H_real_units[H_real_units>0].mean() if np.any(H_real_units>0) else 0
        
        stats_text = (f'Total cells: {active_cells}\n'
                     f'Max density: {max_density:.1f} cells/1000 um^2\n'
                     f'Mean density: {mean_density:.1f} cells/1000 um^2\n'
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