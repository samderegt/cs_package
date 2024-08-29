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

class Gharib_Nezhad_ea_2021_broadening:

    def __init__(self, species='AlH'):
        
        if species in ['AlH']:
            self.a_H2 = [+7.6101e-02, -4.3376e-02, +1.9967e-02, +2.4755e-03]
            self.b_H2 = [-5.6857e-01, +2.7436e-01, +3.6216e-02, +1.5350e-05]
            self.a_He = [+4.8630e-02, +2.1731e+03, -2.5351e+02, +3.8607e+01]
            self.b_He = [+4.4644e+04, -4.4438e+03, +6.9659e+02, +4.7331e+00]

        elif species in ['CaH', 'MgH']:
            self.a_H2 = [+8.4022e-02, -8.2171e+03, +4.6171e+02, -7.9708e+00]
            self.b_H2 = [-9.7733e+04, -1.4141e+03, +2.0290e+02, -1.2797e+01]
            self.a_He = [+4.8000e-02, +7.1656e+02, -3.9616e+01, +6.7367e-01]
            self.b_He = [+1.4992e+04, +1.2361e+02, -1.4988e+01, +1.5056e+00]

        elif species in ['CrH', 'FeH', 'TiH']:
            self.a_H2 = [+7.0910e-02, -6.5083e+04, +2.5980e+03, -3.3292e+01]
            self.b_H2 = [-9.0722e+05, -4.3668e+03, +6.1772e+02, -2.4038e+01]
            self.a_He = [+4.2546e-02, -3.0981e+04, +1.2367e+03, -1.5848e+01]
            self.b_He = [-7.1977e+05, -3.4645e+03, +4.9008e+02, -1.9071e+01]

        elif species in ['SiO']:
            self.a_H2 = [+4.7273e-02, -2.7597e+04, +1.1016e+03, -1.4117e+01]
            self.b_H2 = [-5.7703e+05, -2.7774e+03, +3.9289e+02, -1.5289e+01]
            self.a_He = [+2.8364e-02, -6.7705e+03, +2.7027e+02, -3.4634e+00]
            self.b_He = [-2.3594e+05, -1.1357e+03, +1.6065e+02, -6.2516e+00]

        elif species in ['TiO', 'VO']:
            self.a_H2 = [+1.0000e-01, -2.4549e+05, +8.7760e+03, -8.7104e+01]
            self.b_H2 = [-2.3874e+06, +1.6350e+04, +1.7569e+03, -4.1520e+01]
            self.a_He = [+4.0000e-02, -2.8682e+04, +1.0254e+03, -1.0177e+01]
            self.b_He = [-6.9735e+05, +4.7758e+03, +5.1317e+02, -1.2128e+01]

        else:
            raise ValueError(f"Species \'{species}\' not recognised.")
            
    def Pade_equation(self, J, a, b):
        term1 = a[0] + a[1]*J + a[2]*J**2 + a[3]*J**3
        term2 = 1 + b[0]*J + b[1]*J**2 + b[2]*J**3 + b[3]*J**4
        
        return term1 / term2
    
    def gamma_H2(self, J):
        return self.Pade_equation(J, a=self.a_H2, b=self.b_H2)
    
    def gamma_He(self, J):
        return self.Pade_equation(J, a=self.a_He, b=self.b_He)

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

        # Pressure-broadening parameters
        self.broad = {}
        if hasattr(conf, 'broadening'):
            self._load_pressure_broad(conf.broadening)

    def _load_pressure_broad(self, broadening):

        for species_i, input_i in broadening.items():
            
            # Volume-mixing ratio
            self.broad[species_i] = input_i.copy()

            # Use gamma and T-exponent from input-dictionary
            # gamma * (T0/T)^n * (P/1bar)
            gamma_i = input_i.get('gamma', 0.0)
            if callable(gamma_i):
                self.broad[species_i]['gamma'] = gamma_i
            else:
                self.broad[species_i]['gamma'] = np.array([gamma_i])
            self.broad[species_i]['n'] = np.array([input_i.get('n', 0.0)])

            file_i = input_i.get('file')
            try:
                # Read broadening parameters from file
                br = read_fwf(file_i, header=None)
            except (FileNotFoundError, ValueError) as e:
                # No broadening parameters in file
                continue

            label_i = np.array(br[0], dtype=str)
            gamma_i, n_i = np.array(br[[1,2]]).T

            # Currently only handles these 2 broadening diets
            mask_label = (label_i == 'a0')
            if not mask_label.any():
                mask_label = (label_i == 'm0')
                self.broad[species_i]['label'] = 'm0'

            self.broad[species_i]['gamma'] = gamma_i[mask_label]
            self.broad[species_i]['n']     = n_i[mask_label]

            if (br.shape[0]==1) or (br.shape[1]==3):
                # Single row or no quantum-number columns, 
                # ignore quantum-number dependency
                self.broad[species_i]['gamma'] = np.nanmean(
                    self.broad[species_i]['gamma'], keepdims=True
                    )
                self.broad[species_i]['n'] = np.nanmean(
                    self.broad[species_i]['n'], keepdims=True
                    )
                continue

            # Total angular momentum quantum number
            self.broad[species_i]['J'] = np.array(br[3])[mask_label]
            if br.shape[1] == 4:
                # No (more) quantum numbers in file
                continue

            # TODO: additional quantum numbers
            ## Rotational quantum number
            #self.broad[species_i]['K'] = np.array(br[4])

    def load_final_output(self):
        # Convert to [um], [cm^2/molecule], [bar], [K]
        wave, sigma, P, T = self.load_hdf5_output(self.final_output_file)
        return wave*1e6, sigma*1e4, P*1e-5, T

    def convert_to_pRT2_format(self, out_dir, pRT_wave_file, make_short_file, debug=False):

        print('\nConverting to pRT2 opacity format')

        # Load output
        wave_micron, sigma, P_grid, T_grid = self.load_final_output()
        wave_cm = wave_micron*1e-4

        # Load pRT wavelength-grid
        pRT_wave = np.genfromtxt(pRT_wave_file)

        # Create directory if not exist
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        PTpaths = []

        # Make a nice progress bar
        pbar_kwargs = dict(
            total=len(P_grid)*len(T_grid), 
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', 
        )
        with tqdm(**pbar_kwargs) as pbar:

            # Loop over all PT-points
            for idx_P, P in enumerate(P_grid):
                for idx_T, T in enumerate(T_grid):

                    pRT_file = '{}/sigma_{:.0f}.K_{:.06f}bar.dat'.format(out_dir, T, P)
                    if debug:
                        print(pRT_file)
                    PTpaths.append([f'{P}', f'{T}', pRT_file.split('/')[-1]])
                    
                    # Interpolate onto pRT's wavelength-grid and save
                    interp_func  = interp1d(
                        wave_cm, sigma[:,idx_P,idx_T], 
                        bounds_error=False, fill_value=0.0
                        )
                    interp_sigma = interp_func(pRT_wave)
                    np.savetxt(pRT_file, np.column_stack((pRT_wave, interp_sigma)))
                    
                    pbar.update(1)
        
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
            iterable = tqdm(iterable, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for i in iterable:
            # Check if file exists, or is being written
            tmp_file_i = tmp_file.format(i)
            try:
                # Opacity cross-sections for 1 .trans file
                wave, sigma_i, P, T = self.load_hdf5_output(tmp_file_i)
                # Add to total
                sigma_tot += sigma_i
            except:
                # Move on to next file
                continue

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

        assert('H2_broadening' not in conf.files)
        assert('He_broadening' not in conf.files)
        assert(hasattr(conf, 'broadening'))

        # Instantiate the parent class
        super().__init__(conf)
        print(self.broad)

        # Partition function
        self.Q = np.loadtxt(conf.files['partition_function'])
        
        # Load states-info (ExoMol-specific)
        self.states = self._read_states(conf.files['states'])

    def _read_states(self, file):

        print(f'Reading states from \"{file}\"')

        # How pandas should handle compression
        comp = pathlib.Path(file).suffix.replace('.', '')
        if comp != 'bz2':
            comp = 'infer'
            f = open(file)
        else:
            f = bz2.open(file)

        # Infer column-widths
        col_0 = re.findall('\s+\S+', str(f.readline()))
        col_widths = [len(col) for col in col_0]
        f.close()

        # Load states (ID, E, g, J)
        states = read_fwf(
            file, widths=col_widths[:4], header=None, compression=comp
            )
        states = np.array(states)

        # Check that all states are read
        assert(np.all(np.diff(states[:,0])==1))
        assert(states[0,0]==1)

        return states
    
    def _get_pressure_broad(self, J_lower, J_upper, chunk_size=100_000):

        broad_per_trans = {}
        for species_i, broad_i in self.broad.items():

            broad_per_trans[species_i] = dict(VMR=broad_i['VMR'])

            label_i = broad_i.get('label')
            gamma_i = broad_i['gamma']
            n_i     = broad_i['n']

            if callable(gamma_i):
                # User-provided function
                broad_per_trans[species_i]['gamma'] = gamma_i(J_lower)
            else:
                # Mean gamma if not in broadening table
                broad_per_trans[species_i]['gamma'] = np.nanmean(gamma_i)*np.ones_like(J_lower)

            if callable(n_i):
                # User-provided function
                broad_per_trans[species_i]['n'] = n_i(J_lower)
            else:
                # Mean gamma if not in broadening table
                broad_per_trans[species_i]['n'] = np.nanmean(n_i)*np.ones_like(J_lower)

            #print(gamma_i)
            #print(broad_per_trans[species_i])

            if callable(gamma_i) or callable(n_i):
                continue

            J_i = broad_i.get('J')
            if J_i is None:
                # No quantum-number dependency
                continue
            
            # Read in chunks to avoid memory overload
            for idx_l in range(0, len(J_lower), chunk_size):
                idx_h = min([idx_l+chunk_size,len(J_lower)])

                J_to_match = J_lower[idx_l:idx_h]

                if label_i == 'm0':
                    # Check if transition in R-branch (i.e. lower J quantum 
                    # number is +1 higher than upper state).
                    # In that case, 4th column in .broad is |m|=J_l+1.
                    delta_J = J_lower[idx_l:idx_h]-J_upper[idx_l:idx_h]
                    J_to_match[(delta_J == +1)] += 1
                
                # Indices in .broad table corresponding to each transition
                indices_J = np.argwhere(J_i[None,:]==J_to_match[:,None])

                # Update each transition's broadening parameter
                broad_per_trans[species_i]['gamma'][idx_l+indices_J[:,0]] = gamma_i[indices_J[:,1]]
                broad_per_trans[species_i]['n'][idx_l+indices_J[:,0]]     = n_i[indices_J[:,1]]
                
        return broad_per_trans
    
    def get_cross_sections(self, CS, file, show_pbar=True, debug=False):

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

                    if not line:
                        print(f'Computing for {len(A)} transitions in final chunk')
                    else:
                        print(f'Computing for {len(A)} transitions in chunk')
                        
                    idx_u = np.searchsorted(self.states[:,0], np.array(state_ID_u, dtype=int))
                    idx_l = np.searchsorted(self.states[:,0], np.array(state_ID_l, dtype=int))

                    E_l = self.states[idx_l,1].astype(np.float64)
                    g_u = self.states[idx_u,2]
                    
                    J_l = self.states[idx_l,3].astype(int)
                    J_u = self.states[idx_u,3].astype(int)

                    E_u = self.states[idx_u,1].astype(np.float64)
                    nu_0 = E_u - E_l
                    nu_0 = np.abs(nu_0.astype(np.float64))

                    # Compute line-strength at reference temperature
                    term1 = np.array(A, dtype=np.float64)*g_u / (8*np.pi*(100*sc.c)*nu_0**2)
                    term2 = np.exp(-c2*E_l/CS.T_0) / CS.q_0
                    term3 = (1-np.exp(-c2*nu_0/CS.T_0))
                    S_0 = term1 * term2 * term3

                    # Add to lines to compute at the next iteration
                    nu_0 *= (100*sc.c) # Unit conversion
                    E_l  *= sc.h * (100*sc.c)
                    S_0  *= (100*sc.c)

                    idx_sort = np.argsort(nu_0)
                    nu_0 = nu_0[idx_sort]
                    E_l  = E_l[idx_sort].astype(np.float64)
                    S_0  = S_0[idx_sort].astype(np.float64)

                    # Get J-specific broadening parameters
                    broad_per_trans = self._get_pressure_broad(J_lower=J_l, J_upper=J_u)

                    # Next iteration, compute opacities
                    CS.loop_over_PT_grid(
                        func=CS.get_cross_sections, show_pbar=show_pbar, 
                        nu_0=nu_0, E_low=E_l, S_0=S_0, 
                        broad_per_trans=broad_per_trans, 
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

        assert('H2_broadening' not in conf.files)
        assert('He_broadening' not in conf.files)
        assert(hasattr(conf, 'broadening'))

        # Instantiate the parent class
        super().__init__(conf)
        
        # Remove any quantum-number dependency as we cannot 
        # match ExoMol state IDs with HITRAN/HITEMP
        for species_i, broad_i in self.broad.items():
            self.broad[species_i]['gamma'] = np.nanmean([broad_i.get('gamma', 0.0)])
            self.broad[species_i]['n']     = np.nanmean([broad_i.get('n', 0.0)])
        print(self.broad)

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
                broad_per_trans=self.broad, 
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

    def get_cross_sections(self, CS, file, show_pbar=True):

        print(f'\nComputing cross-sections from \"{file}\"')

        # Read all transitions at once
        with open(file, 'r') as f:
            trans = np.array([
                [
                    float(line[7:23].strip()),
                    float(line[24:36].strip()),
                    float(line[37:44].strip()),
                    float(line[45:51].strip()),
                    float(line[59:65].strip())
                ]
                for line in f.readlines()[2:]  # Skip the header line and column names
                if len(line.strip()) > 65  # Filter out footer or irrelevant lines
            ], dtype=np.float64)

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

        mask_valid = np.ones(len(trans), dtype=bool)
        
        if self.nu_0_to_ignore is not None:
            print('Masking transitions at:')
            # And remove any transitions on request
            for nu_0_i in np.array([self.nu_0_to_ignore]).flatten():
                print(f'{nu_0_i} cm^-1 ({1e4/nu_0_i} um)')
                # Set matching wavenumbers to 'invalid'
                mask_nu_0_i = np.isclose(trans[:,0], nu_0_i)
                mask_valid[mask_nu_0_i] = False

        if None not in [self.E_ion, self.Z]:
            # If alkali (and E_ion/Z given), use Schweitzer+ (1996) vdW
            E_H     = 13.6 * 8065.73 # [cm^-1]
            alpha_H = 0.666793e-24   # [cm^3]
            
            E_low_cm  = E_low/(sc.h*(100*sc.c))    # [cm^-1]
            E_high_cm = E_low_cm + nu_0/(100*sc.c) # [cm^-1]

            # Schweitzer et al. (1996) [cm^6 s^-1]
            for species_i, broad_i in self.broad.items():
                
                alpha_i = broad_i['alpha'] # [cm^3]

                # vdW interaction constant [cm^6 s^-1]
                self.broad[species_i]['C_6'] = np.abs(
                    1.01e-32 * alpha_i/alpha_H * (self.Z+1)**2 * (
                        (E_H/(self.E_ion-E_low_cm))**2 - (E_H/(self.E_ion-E_high_cm))**2
                        )
                )[mask_valid]

        else:
            # If not alkali, use only transitions with valid vdW-damping
            mask_valid = mask_valid & (trans[:,4] != 0.0)

        nu_0  = nu_0[mask_valid]
        E_low = E_low[mask_valid]
        S_0   = S_0[mask_valid]

        gamma_vdW = gamma_vdW[mask_valid]
        log_gamma_N = trans[mask_valid,3]

        # Compute opacities
        CS.loop_over_PT_grid(
            func=CS.get_cross_sections, show_pbar=show_pbar, 
            nu_0=nu_0, E_low=E_low, S_0=S_0, 
            broad_per_trans=self.broad, 
            gamma_vdW=gamma_vdW, 
            log_gamma_N=log_gamma_N
            )
            
        return CS