import numpy as np

from pandas import read_fwf, read_csv
import bz2
import re
import h5py

from scipy.interpolate import interp1d
import scipy.constants as sc

c2 = 1.438777      # cgs: [cm K]
e = 4.80320425e-10 # cgs: [cm^3/2 g^1/2 s^-1]

import pathlib
import wget
import json

from tqdm import tqdm

def load_data(conf):
    if conf.database.lower() in ['hitemp', 'hitran']:
        return HITEMP(conf)
    if conf.database.lower() == 'exomol':
        return ExoMol(conf)
    if conf.database.lower() == 'vald':
        return VALD(conf)

    raise Exception(f'Database \"{conf.database}\" not recognised.')

def wget_if_not_exist(url, out_dir):
    out_name = out_dir + url.split('/')[-1]
    if pathlib.Path(out_name).is_file():
        print(f'File \"{out_name}\" already exists, skipping download')
        return out_name
    return wget.download(url, out=out_dir)

class LineList:
    
    @classmethod
    def load_hdf5_output(cls, file):

        with h5py.File(file, 'r') as f:
            # Make an array [...]
            wave  = f['wave'][...] # [m]
            sigma = 10**f['cross_sec'][...] - 1e-250 # [m^2]
            P = 10**f['P'][...] - 1e-250 # [Pa]
            T = f['T'][...] # [K]

        return wave, sigma, P, T
    
    def __init__(self, conf):

        # Atomic mass [kg]
        self.mass = conf.mass * 1.0e-3/sc.N_A

        self.final_output_file = conf.files['final_output']
        self.N_lines_max = getattr(conf, 'N_lines_max', 10_000_000)

        try:
            # Pressure-broadening parameters
            self.gamma_H2, self.n_H2 = \
                self._read_pressure_broad(conf.files['H2_broadening'])
            self.gamma_He, self.n_He = \
                self._read_pressure_broad(conf.files['He_broadening'])
        except (KeyError, FileNotFoundError) as e:
            self.gamma_H2, self.n_H2 = None, None
            self.gamma_He, self.n_He = None, None
            pass

    def _read_pressure_broad(self, file):

        # Pressure-broadening parameters
        br_file = read_fwf(file)
        br_file = br_file.fillna(value=0)
        
        # gamma * (T0/T)^n * (P/1bar)
        gamma = np.array(br_file)[:,1]
        n     = np.array(br_file)[:,2]

        # Average over all lines
        return np.mean(gamma), np.mean(n)

    def load_final_output(self):
        # Convert to [um], [cm^2/molecule], [bar], [K]
        wave, sigma, P, T = self.load_hdf5_output(self.final_output_file)
        return wave*1e6, sigma*1e4, P*1e-5, T

    def convert_to_pRT2_format(self, out_dir, pRT_wave_file, make_short_file, debug=False):

        # Load output
        wave_micron, sigma, P_grid, T_grid = self.load_final_output()
        wave_cm = wave_micron*1e-4

        # Load pRT wavelength-grid
        pRT_wave = np.genfromtxt(pRT_wave_file)

        # Create directory if not exist
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        PTpaths = []
        for idx_P, P in enumerate(P_grid):
            for idx_T, T in enumerate(T_grid):

                pRT_file = '{}/sigma_{:.0f}.K_{:.06f}bar.dat'.format(out_dir, T, P)
                if debug:
                    print(pRT_file)
                
                PTpaths.append(
                    [f'{P}', f'{T}', pRT_file.split('/')[-1]]
                )
                
                # Interpolate onto pRT's wavelength-grid
                interp_func  = interp1d(
                    wave_cm, sigma[:,idx_P,idx_T], bounds_error=False, fill_value=0.0
                    )
                interp_sigma = interp_func(pRT_wave)

                np.savetxt(
                    pRT_file, np.column_stack((pRT_wave, interp_sigma))
                )
        
        # Create directory if not exist
        short_stream_dir = f'{out_dir}/short_stream/'
        pathlib.Path(short_stream_dir).mkdir(parents=True, exist_ok=True)

        # Save pressures/temperatures corresponding to each file
        np.savetxt(
            f'{short_stream_dir}/PTpaths.ls', np.array(PTpaths), delimiter=' ', fmt='%s'
        )

        # Create pRT input-files
        short_stream_lambs_mass = [
            '# Minimum wavelength in cm', '0.3d-4', 
            '# Maximum wavelength in cm', '28d-4', 
            '# Molecular mass in amu', 
            '{:.3f}d0'.format(self.mass*sc.N_A*1e3), 
        ]
        with open(f'{out_dir}/short_stream_lambs_mass.dat', 'a') as f:
            f.write('\n'.join(short_stream_lambs_mass))

        sigma_list = list(np.array(PTpaths)[:,2])
        with open(f'{out_dir}/sigma_list.ls', 'a') as f:
            f.write('\n'.join(sigma_list))

        molparam_id = [
            '#### Species ID (A2) format', '06', 
            '#### molparam value', '1', 
        ]
        with open(f'{short_stream_dir}/molparam_id.txt', 'a') as f:
            f.write('\n'.join(molparam_id))

        # Copy the make_short.f90 fortran-script over
        import shutil
        shutil.copy(make_short_file, dst=f'{out_dir}/make_short.f90')

        # Make executable and run
        import subprocess
        subprocess.run(['gfortran', '-o', f'{out_dir}/make_short', f'{out_dir}/make_short.f90'])
        subprocess.call('./make_short', cwd=f'{out_dir}')

        # Remove temporary files (on pRT's low-res wavelengths)
        for file_to_remove in np.array(PTpaths)[:,2]:
            file_to_remove = pathlib.Path(f'{out_dir}/{file_to_remove}')
            file_to_remove.unlink()

        if isinstance(self, HITEMP):
            print("\n"+'#'*50)
            print('Database is HITRAN/HITEMP, so line-strengths are scaled by solar isotope ratio.\n\
You may need to change the \"molparam\" value in \"molparam_id.txt\".')
            print('#'*50)

    def combine_cross_sections(self, tmp_file, N_trans, append_to_existing=False):

        print(f'\nCombining temporary cross-sections to file \"{self.final_output_file}\"')
        sigma_tot = 0

        iterable = range(N_trans)
        if N_trans > 1:
            iterable = tqdm(iterable)
        
        for i in iterable:

            # Check if file exists
            tmp_file_i = tmp_file.format(i)
            if not pathlib.Path(tmp_file_i).is_file():
                # Move on to next file
                continue

            # Opacity cross-sections for 1 .trans file
            wave, sigma_i, P, T = self.load_hdf5_output(tmp_file_i)

            # Add to total
            sigma_tot += sigma_i

        if append_to_existing:
            # Load previous output [m], [m^2], [Pa], [K]
            existing_wave, existing_sigma, existing_P, existing_T \
                = self.load_hdf5_output(self.final_output_file)

            # Same wavelength-grid
            assert(len(wave)==len(existing_wave))
            #assert((wave==existing_wave).all())

            # Save in a new output file
            self.final_output_file = \
                self.final_output_file.replace('.hdf5', '_new.hdf5')

            if np.array_equal(existing_P, P):
                # Equal along pressure-axis, insert new temperatures
                is_new_T = ~np.isin(T,existing_T)
                T = np.append(existing_T, T[is_new_T])

                # Add to cross-sections and sort
                sigma_tot = np.append(
                    existing_sigma, sigma_tot[:,:,is_new_T], axis=2
                    )
                sigma_tot = sigma_tot[:,:,np.argsort(T)]
                T = np.sort(T)

            elif np.array_equal(existing_T, T):
                # Equal along temperature-axis, insert new pressures
                is_new_P = ~np.isin(P,existing_P)
                P = np.append(existing_P, P[is_new_P])

                # Add to cross-sections and sort
                sigma_tot = np.append(
                    existing_sigma, sigma_tot[:,is_new_P,:], axis=1
                    )
                sigma_tot = sigma_tot[:,np.argsort(P),:]
                P = np.sort(P)

            else:
                raise ValueError(
                    f'\
Opacity grid should remain rectangular, but file \"{self.final_output_file}\" has:\n\
P_grid: {existing_P}\nT_grid: {existing_T}\nwhile file \"{tmp_file_i}\" has:\n\
P_grid: {P}\nT_grid: {T}.'
                    )

        # Create directory if not exist
        pathlib.Path(self.final_output_file).parent.mkdir(parents=True, exist_ok=True)

        # Save in a single file
        with h5py.File(self.final_output_file, 'w') as f:
            # Rounding to save memory
            dat1 = f.create_dataset(
                'cross_sec', compression='gzip', 
                data=np.around(np.log10(sigma_tot+1e-250),decimals=3)
                )
            dat1.attrs['units'] = 'log(m^2/molecule)'
            
            dat2 = f.create_dataset('wave', compression='gzip', data=wave)
            dat2.attrs['units'] = 'm'

            # Add 1e-250 to allow zero pressure
            dat3 = f.create_dataset('P', compression='gzip', data=np.log10(P+1e-250))
            dat3.attrs['units'] = 'log(Pa)'

            dat4 = f.create_dataset('T', compression='gzip', data=T)
            dat4.attrs['units'] = 'K'

class ExoMol(LineList):

    @classmethod
    def download_data(cls, conf):

        url_def_json = conf.url_def_json
        url_broad = conf.url_broad
        input_dir = conf.input_dir

        # Create destination directory
        pathlib.Path(input_dir).mkdir(parents=True, exist_ok=True)

        # Download the definition file
        def_file = wget_if_not_exist(url_def_json, input_dir)
        url_base = url_def_json.replace('.json', '')
        print()

        # Read definition file
        with open(def_file) as f:
             d = json.load(f)

        # Download partition-function and states files
        print('\nPartition file:')
        partition_file = wget_if_not_exist(f'{url_base}.pf', input_dir)
        print()
        print(pathlib.Path(partition_file).absolute())

        print('\nStates file:')
        states_file = wget_if_not_exist(f'{url_base}.states.bz2', input_dir)
        print()
        print(pathlib.Path(states_file).absolute())
        
        trans_files = []
        print('\nTransition file(s):')
        # Download transition files
        N_trans = d['dataset']['transitions']['number_of_transition_files']
        if N_trans == 1:
            trans_file_i = wget_if_not_exist(f'{url_base}.trans.bz2', input_dir)
            print()
            print(pathlib.Path(trans_file_i).absolute())
        else:
            nu_min = np.linspace(
                0, d['dataset']['transitions']['max_wavenumber'], N_trans, 
                endpoint=False, dtype=int
                )
            delta_nu = nu_min[1]-nu_min[0]

            for nu_min_i in nu_min:
                nu_max_i = nu_min_i + delta_nu

                trans_file_i = wget_if_not_exist(
                    '{}__{:05d}-{:05d}.trans.bz2'.format(url_base, nu_min_i, nu_max_i), 
                    out_dir=input_dir
                    )
                trans_files.append(pathlib.Path(trans_file_i).absolute())
            
            print()
            for trans_file_i in trans_files:
                print(trans_file_i)

        print('\nBroadening files:')
        broad_files = []
        # Download broadening files
        for url_broad_i in url_broad:
            broad_file_i = wget_if_not_exist(url_broad_i, input_dir)
            broad_files.append(pathlib.Path(broad_file_i).absolute())
        
        print()
        for broad_file_i in broad_files:
            print(broad_file_i)
        print()

    def __init__(self, conf):

        # Instantiate the parent class
        super().__init__(conf)

        # Partition function
        self.Q = np.loadtxt(conf.files['partition_function'])
        
        # Load states-info (ExoMol-specific)
        self.states = self._read_states(conf.files['states'])

    def _read_states(self, file):

        print(f'Reading states from \"{file}\"')

        # How pandas should handle compression
        comp = pathlib.Path(file).suffix
        if comp != 'bz2':
            comp = 'infer'

        with bz2.open(file) as f:
            # Infer column-widths
            col_0 = re.findall('\s+\S+', str(f.readline()))
            col_widths = [len(col) for col in col_0]

        # Load states
        states = read_fwf(
            file, widths=col_widths[:4], header=None, compression=comp
            )
        states = np.array(states)

        # Check that all states are read (necessary?)
        assert(np.all(np.diff(states[:,0])==1))
        assert(states[0,0]==1)

        return states
    
    def get_cross_sections(self, CS, file, show_pbar=True, debug=True):

        print(f'\nComputing cross-sections from \"{file}\"')

        i = 0
        state_ID_u = []; state_ID_l = []; A = []

        with bz2.open(file) as f:

            while True:
                # Compute opacities in chunks to prevent memory-overload
                line = f.readline()

                if i == 0:
                    # Infer column-widths in .trans file (only 1st line)
                    sep_line_0 = re.findall('\s+\S+', str(line))
                    col_widths = [len(col) for col in sep_line_0]
                    col_idx = np.cumsum(col_widths)

                elif (i%self.N_lines_max == 0) or (not line):
                    
                    # Compute when N_lines_max have been read, or end-of-file
                    
                    if len(A) == 0:
                        # Last line was included in previous chunk 
                        # (i.e. len(file) == X*N_lines_max)
                        return CS

                    if debug:
                        print('Computing for chunk')

                    if i > self.N_lines_max:
                        print('Computing for final chunk')
                    elif not line:
                        print(f'Computing for {i} transitions')
                        
                    idx_u = np.searchsorted(self.states[:,0], np.array(state_ID_u, dtype=int))
                    idx_l = np.searchsorted(self.states[:,0], np.array(state_ID_l, dtype=int))
                    
                    E_l = self.states[idx_l,1].astype(np.float64)
                    g_u = self.states[idx_u,2]

                    E_u = self.states[idx_u,1].astype(np.float64)
                    nu_0 = E_u - E_l
                    nu_0 = nu_0.astype(np.float64)

                    # Compute line-strength at reference temperature
                    term1 = np.array(A, dtype=np.float64)*g_u / (8*np.pi*(100*sc.c)*nu_0**2)
                    term2 = np.exp(-c2*E_l/CS.T_0) / CS.q_0
                    term3 = (1-np.exp(-c2*nu_0/CS.T_0))
                    S_0 = term1 * term2 * term3
                    
                    # Add to lines to compute at the next iteration
                    nu_0 *= (100*sc.c) # Unit conversion
                    E_l  *= sc.h * (100*sc.c)
                    S_0  *= (100*sc.c)

                    E_l = E_l.astype(np.float64)
                    S_0 = S_0.astype(np.float64)

                    # Next iteration, compute opacities
                    CS.loop_over_PT_grid(
                        func=CS.get_cross_sections, show_pbar=show_pbar, 
                        nu_0=nu_0, E_low=E_l, S_0=S_0, 
                        gamma_H2=self.gamma_H2, n_H2=self.n_H2, 
                        gamma_He=self.gamma_He, n_He=self.n_He, 
                        )
                    
                    if not line:
                        # End-of-file
                        return CS
                    
                    # Empty lists
                    state_ID_u = []; state_ID_l = []; A = []

                # Read transition-info for each 
                # Access info on upper and lower states
                state_ID_u.append(line[0:col_idx[0]])
                state_ID_l.append(line[col_idx[0]:col_idx[1]])
                A.append(line[col_idx[1]:col_idx[2]])

                i += 1

class HITEMP(LineList):

    @classmethod
    def download_data(cls, conf):
        
        urls = conf.urls
        input_dir = conf.input_dir

        # Create destination directory
        pathlib.Path(input_dir).mkdir(parents=True, exist_ok=True)

        files = []
        for url_i in urls:
            file_i = wget_if_not_exist(url_i, input_dir)
            files.append(pathlib.Path(file_i).absolute())

        print()
        for file_i in files:
            print(file_i)
        print()

    def __init__(self, conf):

        # Instantiate the parent class
        super().__init__(conf)

        # Partition function
        self.Q = np.loadtxt(conf.files['partition_function'])

        # Isotope index (HITEMP/HITRAN-specific)
        self.isotope_idx = getattr(conf, 'isotope_idx', 1)

    def get_cross_sections(self, CS, file, show_pbar=True):

        print(f'\nComputing cross-sections from \"{file}\"')

        # How pandas should handle compression
        comp = pathlib.Path(file).suffix
        if comp != 'bz2':
            comp = 'infer'

        # Read only N_lines_max lines to prevent memory-overload
        zip_file = read_fwf(
            file, widths=(2,1,12,10,10,5,5,10,4,8), 
            header=None, chunksize=self.N_lines_max, 
            compression=comp
        )
        for trans_i in zip_file:
            trans_i = np.array(trans_i)

            if self.isotope_idx is not None:
                # Only lines from one isotope
                print(f'Reading isotope-index {self.isotope_idx}')
                trans_i = trans_i[trans_i[:,1]==self.isotope_idx]

            # Sort lines by central wavenumber (!! might only work if delta_P==0 !!)
            trans_i = trans_i[np.argsort(trans_i[:,2]),:]

            # Unit conversion
            nu_0  = trans_i[:,2] * (100*sc.c)        # [cm^-1] -> [s^-1]
            E_low = trans_i[:,7] * sc.h * (100*sc.c) # [cm^-1] -> [kg m^2 s^-2] or [J]
            S_0   = trans_i[:,3] * (100*sc.c)        # [cm^-1/(molec. cm^-2)] -> [s^-1/(molec. cm^-2)]
            
            # Compute opacities
            CS.loop_over_PT_grid(
                func=CS.get_cross_sections, show_pbar=show_pbar, 
                nu_0=nu_0, E_low=E_low, S_0=S_0, 
                gamma_H2=self.gamma_H2, n_H2=self.n_H2, 
                gamma_He=self.gamma_He, n_He=self.n_He, 
                delta_P=np.zeros_like(nu_0) # !! for P-dependent shifts !!
                )
        
        return CS

class VALD(LineList):
    
    def __init__(self, conf):

        # Instantiate the parent class
        super().__init__(conf)

        # Compute partition function from state-info
        self._get_partition_from_NIST_states(conf.files['states'])

        # If alkali, provide E_ion and Z for different vdW-broadening
        self.E_ion = getattr(conf, 'E_ion', None) # [cm^-1]
        self.Z     = getattr(conf, 'Z', None)

        # Transition-energies to ignore
        self.nu_0_to_ignore = getattr(conf, 'nu_0_to_ignore', None)

    def _get_partition_from_NIST_states(self, file, T=np.arange(1,5001+1e-6,1)):

        print(f'Reading states from \"{file}\"')

        # Load states
        states = read_csv(
            file, sep='\t', engine='python', 
            header=0, #skipfooter=1
            )
        g = np.array(states['g'])
        E = np.array(states['Level (cm-1)'])

        # (higher) ions beyond this index
        idx_u = np.min(np.argwhere(np.isnan(g)))
        g = g[:idx_u]
        E = E[:idx_u]

        # Partition function
        self.Q = np.sum(
            g[None,:] * np.exp(-c2*E[None,:]/T[:,None]), 
            axis=-1 # Sum over states, keep T-axis
            )
        self.Q = np.concatenate((T[:,None], self.Q[:,None]), axis=-1)
        print(self.Q)

    def get_cross_sections(self, CS, file, show_pbar=True):

        print(f'\nComputing cross-sections from \"{file}\"')

        # Read all transitions at once
        trans = read_fwf(
            file, header=1, skipfooter=6, 
            colspecs=[(7,23),(24,36), (37,44), (45,51),(59,65)], 
            )
        trans = np.array(trans).astype(np.float64)

        # Oscillator strength
        gf = 10**trans[:,2]

        nu_0  = trans[:,0] # [cm^-1]
        E_low = trans[:,1] # [cm^-1]

        # Compute line-strength at reference temperature
        term1 = (gf * np.pi*e**2) / ((1e3*sc.m_e)*(100*sc.c)**2)
        term2 = np.exp(-c2*E_low/CS.T_0) / CS.q_0
        term3 = (1-np.exp(-c2*nu_0/CS.T_0))
        S_0 = term1 * term2 * term3

        # Unit conversion
        nu_0  *= (100*sc.c)        # [cm^-1] -> [s^-1]
        E_low *= sc.h * (100*sc.c) # [cm^-1] -> [kg m^2 s^-2] or [J]
        S_0   *= (100*sc.c)        # [cm^-1/(molec. cm^-2)] -> [s^-1/(molec. cm^-2)]

        # Single vdW-damping given
        gamma_vdW = 10**trans[:,4] # [s^-1 cm^3]

        if None not in [self.E_ion, self.Z]:
            # If alkali (and E_ion/Z given), use Schweitzer+ (1996) vdW
            alpha_H = 0.666793
            alpha_p = 0.806
            E_H     = 13.6 * 8065.73 # [cm^-1]
            
            E_low_cm = E_low/(sc.h*(100*sc.c))     # [cm^-1]
            E_high_cm = E_low_cm + nu_0/(100*sc.c) # [cm^-1]

            # Schweitzer et al. (1996) [cm^6 s^-1]
            CS.C_6 = np.abs(
                1.01e-32 * alpha_p/alpha_H * (self.Z+1)**2 * \
                ((E_H/(self.E_ion-E_low_cm))**2 - (E_H/(self.E_ion-E_high_cm))**2)
            )
            mask_valid = np.ones_like(trans[:,4], dtype=bool)

        else:
            # If not alkali, use only transitions with valid vdW-damping
            mask_valid = (trans[:,4] < 0.0)

        if self.nu_0_to_ignore is not None:
            print('Masking transitions at:')
            # And remove any transitions on request
            for nu_0_i in np.array([self.nu_0_to_ignore]).flatten():
                print(f'{nu_0_i} cm^-1 ({1e4/nu_0_i} um)')
                # Set matching wavenumbers to 'invalid'
                mask_nu_0_i = np.isclose(trans[:,0], nu_0_i)
                mask_valid[mask_nu_0_i] = False

        nu_0  = nu_0[mask_valid]
        E_low = E_low[mask_valid]
        S_0   = S_0[mask_valid]

        gamma_vdW = gamma_vdW[mask_valid]
        log_gamma_N = trans[mask_valid,3]
        if hasattr(CS, 'C_6'):
            CS.C_6 = CS.C_6[mask_valid]

        # Compute opacities
        CS.loop_over_PT_grid(
            func=CS.get_cross_sections, show_pbar=show_pbar, 
            nu_0=nu_0, E_low=E_low, S_0=S_0, gamma_H2=gamma_vdW, 
            log_gamma_N=log_gamma_N
            )
            
        return CS