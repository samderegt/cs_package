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
                c = cmap(j/max([len(idx_P)-1,1]))
                ax_i.plot(
                    self.wave_micron[mask], self.sigma[mask,idx_P_j,idx_T], 
                    c=c, lw=1, ls='-', #alpha=0.7, 
                    label='P = {:.0e} bar'.format(self.P_grid[idx_P_j])
                    )
                
            xscale = 'linear'
            if wave_max-wave_min > 2:
                xscale = 'log'

            ax_i.set(
                xlim=self.wave_range[i], ylim=ylim, yscale='log', xscale=xscale
                )
        ax[-1].legend(title='T = {:.0f} K'.format(self.T_grid[idx_T]))
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
                c = cmap(j/max([len(idx_T)-1,1]))
                ax_i.plot(
                    self.wave_micron[mask], self.sigma[mask,idx_P,idx_T_j], 
                    c=c, lw=1, ls='-', #alpha=0.7, 
                    label='T = {:.0f} K'.format(self.T_grid[idx_T_j])
                    )

            xscale = 'linear'
            if wave_max-wave_min > 2:
                xscale = 'log'

            ax_i.set(
                xlim=self.wave_range[i], ylim=ylim, yscale='log', xscale=xscale
                )
        ax[-1].legend(title='P = {:.0e} bar'.format(self.P_grid[idx_P]))
        ax[-1].set(
            xlabel=r'Wavelength ($\mu$m)', ylabel=r'Cross-section (cm$^2$/molecule)'
            )
        plt.tight_layout()
        plt.savefig(save_file)
        plt.close()

class pRT_Figures:

    def __init__(self, line_species=[], wave_range=None):

        # Handling multiple line_species
        self.line_species = np.array([line_species]).flatten()

        # Create multiple subplots for multiple wavelength-ranges
        self.wave_range = wave_range
        self.nrows = 1
        if isinstance(self.wave_range, list):
            self.nrows = len(self.wave_range)
        else:
            self.wave_range = [self.wave_range]

        wave_min = np.min(np.array(self.wave_range))
        wave_max = np.max(np.array(self.wave_range))

        # Create pRT Radtrans objects
        from petitRADTRANS import Radtrans
        self.atm = [
            Radtrans(
                line_species=[line_species_i], mode='lbl', 
                wlen_bords_micron=[wave_min,wave_max], 
                #lbl_opacity_sampling=5
                )
            for line_species_i in self.line_species
        ]

    def _get_opacities(self, P, T):

        opa = []
        for atm_i in self.atm:
            wave_micron, opa_i = atm_i.plot_opas(
                atm_i.line_species, temperature=T, pressure_bar=P, return_opacities=True
                )[atm_i.line_species[0]]
            opa.append(opa_i)

        return wave_micron, opa

    def _get_ls(cls, idx):
        ls = ['-', '--', '-.', ':']
        if idx >= len(ls):
            idx %= len(ls)
        return ls[idx]

    def plot_P(self, T=1000, P=np.logspace(-2,2,5), cmap='viridis', save_file='plots/pRT_P.pdf', ylim=None):

        cmap = plt.get_cmap(cmap)
        
        # Make (multi-row) figure
        fig, ax = plt.subplots(
            figsize=(12,2.8*self.nrows), nrows=self.nrows
            )
        if self.nrows == 1:
            ax = [ax]
        
        
        for i, P_i in enumerate(P):

            # Get opacity cross-sections
            wave_micron, opa = self._get_opacities(P=P_i, T=T)
            c = cmap(i/max([len(P)-1,1]))

            for j, ax_j in enumerate(ax):    

                # Only plot wavelength-range in this subplot
                wave_min, wave_max = self.wave_range[j]
                mask = (wave_micron>=wave_min-0.01) & \
                    (wave_micron<=wave_max+0.01)
                
                for k, opa_k in enumerate(opa):
                    ls = self._get_ls(k)

                    label = None
                    if k == 0:
                        label='P = {:.0e} bar'.format(P_i)

                    ax_j.plot(
                        wave_micron[mask], opa_k[mask], c=c, 
                        lw=1, ls=ls, label=label
                        )
                
                if i > 0:
                    continue

                xscale = 'linear'
                if wave_max-wave_min > 2:
                    xscale = 'log'

                ax_j.set(yscale='log', xscale=xscale)
                ax_j.set(xlim=self.wave_range[j], ylim=ylim)
                
        ax[-1].legend(title='T = {:.0f} K'.format(T))
        ax[-1].set(
            xlabel=r'Wavelength ($\mu$m)', ylabel=r'Cross-section (cm$^2$/g)'
            )
        plt.tight_layout()
        plt.savefig(save_file, dpi=200)
        plt.close()

    def plot_T(self, P=1, T=np.arange(500,2500,500), cmap='coolwarm', save_file='plots/pRT_T.pdf', ylim=None):

        cmap = plt.get_cmap(cmap)
        
        # Make (multi-row) figure
        fig, ax = plt.subplots(
            figsize=(12,2.8*self.nrows), nrows=self.nrows
            )
        if self.nrows == 1:
            ax = [ax]
        
        for i, T_i in enumerate(T):

            # Get opacity cross-sections
            wave_micron, opa = self._get_opacities(P=P, T=T_i)
            c = cmap(i/max([len(T)-1,1]))

            for j, ax_j in enumerate(ax):    

                # Only plot wavelength-range in this subplot
                wave_min, wave_max = self.wave_range[j]
                mask = (wave_micron>=wave_min-0.01) & \
                    (wave_micron<=wave_max+0.01)
                
                for k, opa_k in enumerate(opa):
                    ls = self._get_ls(k)

                    label = None
                    if k == 0:
                        label='T = {:.0f} K'.format(T_i)

                    ax_j.plot(
                        wave_micron[mask], opa_k[mask], c=c, 
                        lw=1, ls=ls, label=label
                        )
                
                if i > 0:
                    continue

                xscale = 'linear'
                if wave_max-wave_min > 2:
                    xscale = 'log'

                ax_j.set(yscale='log', xscale=xscale)
                ax_j.set(xlim=self.wave_range[j], ylim=ylim)
                
        ax[-1].legend(title='P = {:.0e} bar'.format(P))
        ax[-1].set(
            xlabel=r'Wavelength ($\mu$m)', ylabel=r'Cross-section (cm$^2$/g)'
            )
        plt.tight_layout()
        plt.savefig(save_file, dpi=200)
        plt.close()

if __name__ == '__main__':
    #wave_min, wave_max = 2.0, 2.2
    wave_min, wave_max = 1.1, 1.35
    #wave_min, wave_max = 2.05, 2.47
    #wave_min, wave_max = 2.05, 2.23
    #wave_min, wave_max = 2.23, 2.47
    wave_range = [
        (wave_i, wave_i+0.005) \
        for wave_i in np.arange(wave_min, wave_max, 0.005)
    ]
    F = pRT_Figures(
        #line_species='15NH3_CoYuTe', 
        #wave_range=[(1.9,2.5), (2.0,2.01), (2.3,2.31)], 
        #line_species=['H2O_pokazatel_main_iso_Sam', 'H2O_pokazatel_main_iso'], 
        line_species=['CH4_hargreaves_main_iso', 'tmp_CH4_MM_main_iso'], 
        #line_species=['CO_36_high_Sam', 'CO_36_high'], 
        #line_species=['CO_high_Sam', 'CO_high'], 
        wave_range=wave_range, 
        )
    #F.plot_P(P=np.array([0.1]))
    F.plot_T(T=np.array([1200.]), ylim=(1e0,1e1))#, save_file='plots/pRT_T.png')