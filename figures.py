import numpy as np
import matplotlib.pyplot as plt

import pathlib

def find_nearest(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        idx = np.abs(a[:,None]-b[None,:]).argmin(axis=0)
    else:
        idx = np.abs(a-b).argmin()
    return idx, a[idx]
    
class Figures:
        
    def __init__(self, D, wave_range=None):
        
        # Load output
        self.wave_micron, self.sigma, self.P_grid, self.T_grid \
            = D.load_final_output()
        
        # Create multiple subplots for multiple wavelength-ranges
        self.wave_range = wave_range
        self.nrows = 1
        if isinstance(self.wave_range, list):
            self.nrows = len(self.wave_range)
        else:
            self.wave_range = [self.wave_range]

    def plot_P(self, T=1000, P=None, cmap='viridis', save_file='plots/P.pdf', ylim=None):

        pathlib.Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Indices to plot
        if P is None:
            idx_P = np.arange(len(self.P_grid))
        else:
            idx_P, _ = find_nearest(
                np.log10(self.P_grid), np.sort(np.log10(P))
                )
        idx_T, _ = find_nearest(self.T_grid, T)

        cmap = plt.get_cmap(cmap)
        
        # Make (multi-row) figure
        fig, ax = plt.subplots(
            figsize=(12,2.8*self.nrows), nrows=self.nrows
            )
        if self.nrows == 1:
            ax = [ax]

        for i, ax_i in enumerate(ax):

            # Only plot wavelength-range in this subplot
            wave_min, wave_max = self.wave_range[i]
            mask = (self.wave_micron>=wave_min-0.01) & \
                (self.wave_micron<=wave_max+0.01)
            
            for j, idx_P_j in enumerate(idx_P):
                c = cmap(j/(len(idx_P)-1))
                ax_i.plot(
                    self.wave_micron[mask], self.sigma[mask,idx_P_j,idx_T], 
                    c=c, lw=1, ls='-', #alpha=0.7, 
                    label='P = {:.0e} bar'.format(self.P_grid[idx_P_j])
                    )

            ax_i.set(
                xlim=self.wave_range[i], ylim=ylim, yscale='log'
                )
        ax[-1].legend()
        ax[-1].set(
            xlabel=r'Wavelength ($\mu$m)', ylabel=r'Cross-section (cm$^2$/molecule)'
            )
        plt.tight_layout()
        plt.savefig(save_file)
        plt.close()

    def plot_T(self, P=1, T=None, cmap='coolwarm', save_file='plots/T.pdf', ylim=None):

        pathlib.Path(save_file).parent.mkdir(parents=True, exist_ok=True)

        # Indices to plot
        if T is None:
            idx_T = np.arange(len(self.T_grid))
        else:
            idx_T, _ = find_nearest(self.T_grid, np.sort(T))
        idx_P, _ = find_nearest(self.P_grid, P)

        cmap = plt.get_cmap(cmap)
        
        # Make (multi-row) figure
        fig, ax = plt.subplots(
            figsize=(12,2.8*self.nrows), nrows=self.nrows
            )
        if self.nrows == 1:
            ax = [ax]

        for i, ax_i in enumerate(ax):

            # Only plot wavelength-range in this subplot
            wave_min, wave_max = self.wave_range[i]
            mask = (self.wave_micron>=wave_min-0.01) & \
                (self.wave_micron<=wave_max+0.01)
            
            for j, idx_T_j in enumerate(idx_T):
                c = cmap(j/(len(idx_T)-1))
                ax_i.plot(
                    self.wave_micron[mask], self.sigma[mask,idx_P,idx_T_j], 
                    c=c, lw=1, ls='-', #alpha=0.7, 
                    label='T = {:.0f} K'.format(self.T_grid[idx_T_j])
                    )

            ax_i.set(
                xlim=self.wave_range[i], ylim=ylim, yscale='log'
                )
        ax[-1].legend()
        ax[-1].set(
            xlabel=r'Wavelength ($\mu$m)', ylabel=r'Cross-section (cm$^2$/molecule)'
            )
        plt.tight_layout()
        plt.savefig(save_file)
        plt.close()
