import numpy as np
import scipy.constants as sc
from scipy.special import wofz
import h5py

import pathlib
from tqdm import tqdm

class CrossSection:

    @classmethod
    def _apply_mask(cls, list_of_arrays, cond):
        return [el[cond] for el in list_of_arrays]

    def __init__(self, conf, Q, mass):

        self.contains_lines = False

        self.Q = Q
        self.mass = mass

        self.T_0 = getattr(conf, 'T_0', 296.)
        self.q_0 = np.interp(self.T_0, self.Q[:,0], self.Q[:,1])

        # Create wavenumber grid
        self._set_nu_grid(conf)
        
        # (P,T)-grid to compute cross-sections on
        self.P_grid = np.asarray(conf.P_grid) * 1e5 # [bar] -> [Pa]
        self.T_grid = np.asarray(conf.T_grid)
        self.N_PT = len(self.P_grid) * len(self.T_grid)

        # Cross-sections array
        self.sigma = np.zeros(
            (len(self.nu_grid), len(self.P_grid), len(self.T_grid))
            )

        # Line-wing cutoff
        self.wing_cutoff = getattr(
            conf, 'wing_cutoff', 
            lambda _, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024)
            )
        #self.cutoff_max = getattr(conf, 'cutoff_max', 25)
        self.cutoff_max = getattr(conf, 'cutoff_max', np.inf)

        # Line-strength cutoffs
        self.local_cutoff  = getattr(conf, 'local_cutoff', 0.35)
        self.global_cutoff = getattr(conf, 'global_cutoff', 1e-45)

        # Print some info
        print()
        print('Wavelength range (um):              ', 1e6*sc.c/self.nu_grid[[-1,0]])
        print('Number of points on wavelength grid:', len(self.nu_grid))
        print('Wavenumber spacing (cm^-1):         ', np.mean(np.diff(self.nu_grid))/(100.0*sc.c))
        print()
        print('P (bar):', self.P_grid*1e-5)
        print('T (K):  ', self.T_grid)
        print()
        pass

    def _set_nu_grid(self, conf):

        # Pre-computed wavelength-grid
        wave_file = conf.files.get('wavelength')
        if wave_file is not None:
            raise NotImplementedError('Custom wavelength-grids are not implemented yet.')
            
            # Wavelengths in [um]
            wave = np.loadtxt(wave_file)

            # TODO: Not sure this is the same units as Sid's?
            # Varying delta_nu's also won't work

            # Wavenumbers in [cm^-1]
            self.nu_grid  = 1e4/wave
            self.delta_nu = np.diff(self.nu_grid)
            self.N_grid   = len(self.nu_grid)
            return
        
        # Create new wavenumber grid
        self.delta_nu = conf.delta_nu # [cm^-1]
        self.delta_nu *= 100.0*sc.c # [cm^-1] -> [s^-1]

        self.wave_min = conf.wave_min*1e-6 # [um] -> [m]
        self.wave_max = conf.wave_max*1e-6 # [um] -> [m]

        self.nu_min = sc.c/self.wave_max # [m] -> [s^-1]
        self.nu_max = sc.c/self.wave_min # [m] -> [s^-1]

        # Number of grid points
        self.N_grid   = int((self.nu_max-self.nu_min)/self.delta_nu) + 1

        # Not exact value of delta_nu given above, but done to keep final lambda values fixed
        self.delta_nu = (self.nu_max-self.nu_min) / (self.N_grid-1)
        self.nu_grid  = np.linspace(
            self.nu_min, self.nu_max, num=self.N_grid, endpoint=True
        )

    def _line_strength(self, T, S_0, E_low, nu_0):

        # Partition function
        q = np.interp(T, self.Q[:,0], self.Q[:,1])

        # Use polynomial extrapolation for T outside range
        if T > self.Q[-1,0]:
            p = np.polyfit(self.Q[-5:,0], self.Q[-5:,1], deg=3) # Last 5 points
            q = np.poly1d(p)(T)
        elif T < self.Q[0,0]:
            p = np.polyfit(self.Q[:5,0], self.Q[:5,1], deg=3) # First 5 points
            q = np.poly1d(p)(T)

        # Gordon et al. (2017) (E_low: [cm^-1]; nu_0: [s^-1])
        term1 = S_0 * (self.q_0/q) * np.exp(E_low/sc.k*(1/self.T_0-1/T))
        term2 = (1-np.exp(-sc.h*nu_0/(sc.k*T))) / (1-np.exp(-sc.h*nu_0/(sc.k*self.T_0)))
        return term1 * term2

    def _gamma_vdW(
            self, P, T, gamma_H2, n_H2, gamma_He=None, n_He=None, VMR_H2=0.85, VMR_He=0.15
            ):

        if gamma_He is not None:
            gamma_H2 *= (self.T_0/T)**n_H2 * (P*1e-5) * (100*sc.c) * VMR_H2 # [cm^-1] -> [s^-1]
            gamma_He *= (self.T_0/T)**n_He * (P*1e-5) * (100*sc.c) * VMR_He
            return gamma_H2 + gamma_He

        # Number densities
        N_tot = P / (sc.k*T) * (100)**(-3) # [cm^-3]
        N_H2 = N_tot * VMR_H2
        N_He = N_tot * VMR_He

        if hasattr(self, 'C_6'):
            # Is alkali line, use different vdW-damping
            mass_pert = 2.016*sc.m_u # [kg]

            # Schweitzer et al. (1996) (omitting 1/c to get [s^-1])
            gamma_vdW = 1.664461/2 * (sc.k*T * (1/self.mass+1/mass_pert))**(0.3) * \
                self.C_6**(2/5) * N_tot # [s^-1]
            return gamma_vdW
            
        # Only one vdW-damping is given (from Kurucz)
        C_H2, C_He = 0.85, 0.42 # Diff. polarisability (Kurucz & Furenlid 1979)

        # Sharp & Burrows (2007) (omitting 1/c to get [s^-1])
        gamma_vdW = gamma_H2/(4*np.pi) * (T/10000)**(0.3) * (C_H2*N_H2 + C_He*N_He) # [s^-1]
        return gamma_vdW
    
    def _gamma_N(self, nu_0, log_gamma_N=None):

        gamma_N = 0.222 * (nu_0/(sc.c*100.0))**2 / (4.0*np.pi*sc.c) # [s^-1]

        if log_gamma_N is not None:
            # Radiative/natural damping constant is given (from Kurucz)
            mask_valid = (log_gamma_N != 0)
            gamma_N[mask_valid] = 10**log_gamma_N[mask_valid] / (4*np.pi) # [s^-1]
        
        return gamma_N

    def _gamma_G(self, T, nu_0):
        return np.sqrt(2*sc.k*T/self.mass) * nu_0/sc.c # [s^-1]
    
    def _local_cutoff(self, S, nu_0, factor):

        # Round to zero-th decimal
        nu_bin = np.around((nu_0-self.nu_min)/self.delta_nu).astype(int)

        # Upper and lower indices of lines within bins
        _, nu_bin_idx = np.unique(nu_bin, return_index=True)
        nu_bin_idx    = np.append(nu_bin_idx, len(nu_bin))

        for k in range(len(nu_bin_idx)-1):
            # Cumulative sum of lines in bin
            S_range = S[nu_bin_idx[k]:nu_bin_idx[k+1]]
            S_sort  = np.sort(S_range)
            S_summed = np.cumsum(S_sort)

            # Lines contributing less than 'factor' to total strength
            i_search = np.searchsorted(S_summed, factor*S_summed[-1])
            S_cutoff = S_sort[i_search]

            # Add weak line-strengths to strongest line
            sum_others = np.sum(S_range[S_range<S_cutoff])
            S_range[np.argmax(S_range)] += sum_others

            # Ignore weak lines
            S_range[S_range<S_cutoff] = 0.

            S[nu_bin_idx[k]:nu_bin_idx[k+1]] = S_range

        return S
    
    def _coarse_nu_grid(self, delta_nu_coarse):
        
        # Decrease number of points in wavenumber grid
        coarse_inflation = delta_nu_coarse / self.delta_nu
        N_grid_coarse    = int(self.N_grid/coarse_inflation)

        # Expand wavenumber grid slightly
        delta_nu_ends = delta_nu_coarse * (coarse_inflation-1)/2
        nu_grid_coarse = np.linspace(
            self.nu_min-delta_nu_ends, self.nu_max+delta_nu_ends, 
            num=N_grid_coarse, endpoint=True
        )
        return nu_grid_coarse
    
    def _fast_line_profiles(
            self, nu_0, S, gamma_L, gamma_G, 
            nu_grid, nu_line, idx_to_insert, 
            cutoff_dist_n, chunk_size=200, 
            ):
        
        sigma = np.zeros_like(nu_grid)
        a = gamma_L / gamma_G # Gandhi et al. (2020)

        # Only consider 'chunk_size' lines at a time
        N_chunks = int(np.ceil(len(S)/chunk_size))

        for ch in range(N_chunks):
            
            # Upper and lower indices of lines in current chunk
            idx_ch_l = int(ch*chunk_size)
            idx_ch_h = idx_ch_l + chunk_size
            idx_ch_h = np.minimum(idx_ch_h, len(S)) # At last chunk

            # Indices of nu_grid_coarse to insert current lines in
            idx_to_insert_ch = idx_to_insert[idx_ch_l:idx_ch_h]

            # Lines in current chunk | (N_lines,1)
            nu_0_ch    = nu_0[idx_ch_l:idx_ch_h,None]
            gamma_G_ch = gamma_G[idx_ch_l:idx_ch_h,None]
            a_ch = a[idx_ch_l:idx_ch_h,None]
            S_ch = S[idx_ch_l:idx_ch_h,None]

            # Correct for coarse grid | (N_lines,1)
            nu_grid_ch = nu_grid[idx_to_insert_ch,None]

            # Eq. 10 (Gandhi et al. 2020) | (N_lines,N_wave[cut])
            u = (nu_line[None,:]+nu_grid_ch - nu_0_ch) / gamma_G_ch

            # (Scaled) Faddeeva function for Voigt profiles | (N_lines,N_wave[cut])
            sigma_ch = S_ch * np.real(wofz(u+a_ch*1j)) / (gamma_G_ch*np.sqrt(np.pi))

            # Upper and lower index of these lines in sigma_coarse
            s_l = np.maximum(0, idx_to_insert_ch-cutoff_dist_n+1)
            s_h = np.minimum(len(nu_grid), idx_to_insert_ch+cutoff_dist_n)

            # Upper and lower index of these lines in sigma_ch
            f_l = np.maximum(0, cutoff_dist_n-1-idx_to_insert_ch)
            f_h = 2*cutoff_dist_n - 1 - np.maximum(0,idx_to_insert_ch+cutoff_dist_n-len(nu_grid))

            # Loop over each line profile
            for i, sigma_i in enumerate(sigma_ch):
                # Add line to total cross-section
                sigma[s_l[i]:s_h[i]] += sigma_i[f_l[i]:f_h[i]]

        return sigma

    def loop_over_PT_grid(self, func, show_pbar=True, **kwargs):
        
        # Make a nice progress bar
        pbar_kwargs = dict(
            total=self.N_PT, disable=(not show_pbar), 
            bar_format='{l_bar}{bar:8}{r_bar}{bar:-8b}', 
        )
        with tqdm(**pbar_kwargs) as pbar:

            # Loop over all PT-points
            for idx_P, P in enumerate(self.P_grid):
                for idx_T, T in enumerate(self.T_grid):

                    pbar.set_postfix(P='{:.0e} bar'.format(P*1e-5), T='{:.0f} K'.format(T), refresh=False)
                    func(idx_P, P, idx_T, T, **kwargs)
                    pbar.update(1)

    def get_cross_sections(
            self, 
            idx_P, P, idx_T, T, 
            nu_0, E_low, S_0, 
            gamma_H2, n_H2=None, gamma_He=None, n_He=None, 
            log_gamma_N=None, 
            delta_P=None, 
            debug=False, 
            **kwargs
            ):
        
        if not isinstance(gamma_H2, np.ndarray):
            # Expand to array for each line
            gamma_H2 *= np.ones_like(nu_0)
            
            if n_H2 is not None:
                n_H2 *= np.ones_like(nu_0)

            if gamma_He is not None:
                gamma_He *= np.ones_like(nu_0)
                n_He     *= np.ones_like(nu_0)

        if delta_P is not None:
            # P-dependent shifts (HITRAN/HITEMP)
            nu_0 = nu_0 + delta_P*(100*sc.c)*(P*1e-5)

            # Sort by the new wavenumbers
            idx_sort = np.argsort(nu_0)
            
            nu_0  = nu_0[idx_sort]
            E_low = E_low[idx_sort]
            S_0   = S_0[idx_sort]

            gamma_H2 = gamma_H2[idx_sort]
            if n_H2 is not None:
                n_H2 = n_H2[idx_sort]

            if gamma_He is not None:
                gamma_He = gamma_He[idx_sort]
                n_He     = n_He[idx_sort]
    
        # Get line-strengths and -widths
        S = self._line_strength(T, S_0, E_low, nu_0)
        gamma_L = self._gamma_N(nu_0, log_gamma_N) + \
            self._gamma_vdW(P, T, gamma_H2, n_H2, gamma_He, n_He)
        gamma_G = self._gamma_G(T, nu_0)

        # Select only lines within nu_grid
        nu_0, S, gamma_L, gamma_G = self._apply_mask(
            [nu_0,S,gamma_L,gamma_G], nu_0<self.nu_max
            )
        nu_0, S, gamma_L, gamma_G = self._apply_mask(
            [nu_0,S,gamma_L,gamma_G], nu_0>self.nu_min
            )
        if len(S)==0:
            # No more lines
            return
        
        # Local + global line-strength cutoffs
        if self.local_cutoff is not None:
            S = self._local_cutoff(S, nu_0, factor=self.local_cutoff)
            nu_0, S, gamma_L, gamma_G = self._apply_mask(
                [nu_0,S,gamma_L,gamma_G], S>0
            )
        if self.global_cutoff is not None:
            nu_0, S, gamma_L, gamma_G = self._apply_mask(
                [nu_0,S,gamma_L,gamma_G], S>self.global_cutoff
                )
        if len(S)==0:
            # No more lines
            return
        
        self.contains_lines = True

        if debug:
            print('Number of lines:', len(S))

        # Voigt width
        gamma_V = 0.5346*gamma_L+np.sqrt(0.2166*gamma_L**2 + gamma_G**2)

        # Change to coarse wavenumber grid if lines are wide
        delta_nu_coarse = np.mean(gamma_V) / 6
        if delta_nu_coarse > self.delta_nu:
            nu_grid_coarse = self._coarse_nu_grid(delta_nu_coarse)
        else:
            nu_grid_coarse = self.nu_grid

        delta_nu_coarse = nu_grid_coarse[1] - nu_grid_coarse[0]

        # Indices where lines should be inserted
        idx_to_insert = np.searchsorted(nu_grid_coarse, nu_0) - 1
        
        # TODO: not ideal when Voigt-widths are different per line?
        # Wing cutoff [cm^-1] from given lambda-function
        cutoff = self.wing_cutoff(
            np.mean(gamma_V)/(100*sc.c), # [s^-1] -> [cm^-1]
            P*1e-5 # [Pa] -> [bar]
            )
        cutoff = np.minimum(cutoff, self.cutoff_max) # Use shortest cutoff-distance

        # Number of grid points to consider (from line-center)
        cutoff_dist_n = int(np.around(cutoff*(100*sc.c)/delta_nu_coarse) + 2)

        # Array of wavenumbers from line-center
        nu_line = np.arange(cutoff_dist_n) * delta_nu_coarse
        nu_line = np.concatenate((-nu_line[::-1], nu_line[1:])) # Make symmetric

        # Compute chunk_size line-profiles at once with fast indexing
        sigma_coarse = self._fast_line_profiles(
            nu_0, S, gamma_L, gamma_G, 
            nu_grid=nu_grid_coarse, 
            nu_line=nu_line, 
            idx_to_insert=idx_to_insert, 
            cutoff_dist_n=cutoff_dist_n, 
            chunk_size=200, 
            )

        # Add to total cross-sections
        if len(nu_grid_coarse) == len(self.nu_grid):
            self.sigma[:,idx_P,idx_T] += sigma_coarse
            return

        # Interpolate to original wavenumber grid
        self.sigma[:,idx_P,idx_T] += np.interp(
            self.nu_grid, nu_grid_coarse, sigma_coarse
            )
        return

    def save_cross_sections(self, file):

        print(f'\nSaving cross-sections to file \"{file}\"')
        if not self.contains_lines:
            # No point in saving an array of 0's
            print('No lines in current cross-sections, not saving')
            return

        # Create directory if not exist
        pathlib.Path(file).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(file, 'w') as f:
            # Flip arrays to be ascending in wavelength
            wave = sc.c / self.nu_grid[::-1]
            cross_sec = self.sigma[::-1,:,:]

            f.create_dataset(
                'cross_sec', compression='gzip', 
                data=np.log10(cross_sec*1e-4 + 1e-250) # [cm^2/molecule] -> log10([m^2/molecule])
                )
            f.create_dataset('wave', compression='gzip', data=wave) # [m]
            
            # Add 1e-250 to allow zero pressure
            f.create_dataset('P', compression='gzip', data=np.log10(self.P_grid+1e-250)) # [log10(Pa)]
            f.create_dataset('T', compression='gzip', data=self.T_grid) # [K]