import numpy as np

from scipy.interpolate import interp1d
import scipy.constants as sc

from tqdm import tqdm
import h5py

import pathlib
parent_dir = pathlib.Path(__file__).parent.resolve()

def convert_to_pRT2_format(conf, Data, debug=False, ncpus=1):

    print('\nConverting to pRT2 opacity format')

    # Load output
    wave_micron, sigma, P_grid, T_grid = Data.load_final_output()
    wave_cm = wave_micron * 1e-4

    # Load pRT wavelength-grid
    pRT_wave_file = conf.files.get(
        'pRT_wave', f'{parent_dir}/input_data/wlen_petitRADTRANS.dat'
        )
    pRT_wave = np.genfromtxt(pRT_wave_file)

    # Create directory if not exist
    pRT2_output_dir = f'{Data.output_dir}/pRT2/'
    pathlib.Path(pRT2_output_dir).mkdir(parents=True, exist_ok=True)

    PTpaths = []

    # Prepare the list of tasks
    tasks = []
    for idx_P, P in enumerate(P_grid):
        for idx_T, T in enumerate(T_grid):
            sigma_PT = sigma[:, idx_P, idx_T]
            tasks.append((P, T, wave_cm, sigma_PT, pRT_wave, pRT2_output_dir, debug))

    # Make a nice progress bar
    pbar_kwargs = dict(
        total=len(tasks),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
    )

    if ncpus > 1:
        # Use multiprocessing with the specified number of CPUs
        import multiprocessing as mp
        with mp.Pool(ncpus) as pool:
            with tqdm(**pbar_kwargs) as pbar:
                for result in pool.starmap(process_PT_point, tasks):
                    PTpaths.append(result)
                    pbar.update(1)
    else:
        # Run sequentially without multiprocessing
        with tqdm(**pbar_kwargs) as pbar:
            for task in tasks:
                result = process_PT_point(*task)
                PTpaths.append(result)
                pbar.update(1)

    # Create directory if not exist
    short_stream_dir = f'{pRT2_output_dir}/short_stream/'
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
        '{:.3f}d0'.format(Data.mass*sc.N_A*1e3), 
    ]
    with open(f'{pRT2_output_dir}/short_stream_lambs_mass.dat', 'w') as f:
        f.write('\n'.join(short_stream_lambs_mass))

    sigma_list = list(np.array(PTpaths)[:,2])
    with open(f'{pRT2_output_dir}/sigma_list.ls', 'w') as f:
        f.write('\n'.join(sigma_list))

    molparam_id = [
        '#### Species ID (A2) format', '06', 
        '#### molparam value', '1', 
    ]
    with open(f'{short_stream_dir}/molparam_id.txt', 'w') as f:
        f.write('\n'.join(molparam_id))

    # Copy the make_short.f90 fortran-script over
    import shutil
    make_short_file = conf.files.get(
        'pRT_make_short', f'{parent_dir}/input_data/make_short.f90'
        )
    shutil.copy(make_short_file, dst=f'{pRT2_output_dir}/make_short.f90')

    # Make executable and run
    import subprocess
    subprocess.run(
        ['gfortran', '-o', f'{pRT2_output_dir}/make_short', f'{pRT2_output_dir}/make_short.f90']
        )
    subprocess.call('./make_short', cwd=f'{pRT2_output_dir}')

    # Remove temporary files (on pRT's low-res wavelengths)
    for file_to_remove in np.array(PTpaths)[:,2]:
        file_to_remove = pathlib.Path(f'{pRT2_output_dir}/{file_to_remove}')
        file_to_remove.unlink()

    if Data.database == 'hitemp':
        print('\n'+'#'*50)
        print('Database is HITRAN/HITEMP, so line-strengths are scaled by solar isotope ratio.\n\
You may need to change the \"molparam\" value in \"molparam_id.txt\".')
        print('#'*50)

def process_PT_point(P, T, wave_cm, sigma_PT, pRT_wave, out_dir, debug=False):
    """
    Helper function to parallelize the interpolation to pRT2 and saving for each (P,T) point.
    """
    pRT_file = f'{out_dir}/sigma_{T:.0f}.K_{P:.06f}bar.dat'
    if debug:
        print(pRT_file)

    # Interpolate onto pRT's wavelength-grid
    interp_func = interp1d(wave_cm, sigma_PT, bounds_error=False, fill_value=0.0)
    interp_sigma = interp_func(pRT_wave)

    # Save the result to the file
    np.savetxt(pRT_file, np.column_stack((pRT_wave, interp_sigma)))

    return P, T, pRT_file.split('/')[-1]


def convert_to_pRT3_format(conf, Data, debug=False, **kwargs):

    print('\nConverting to pRT3 opacity format')

    # Load output: wave [um], sigma [cm^2/molecule], P [bar], T [K]
    wave_um, sigma, P_grid, T_grid = Data.load_final_output()
    wave_cm = wave_um * 1e-4

    # Load pRT wavelength-grid
    pRT_wave_file = conf.files.get(
        'pRT_wave', f'{parent_dir}/input_data/wlen_petitRADTRANS.dat'
        )
    pRT_wave = np.genfromtxt(pRT_wave_file) # [cm]
    if kwargs.get('crop_to_28um', False): # Set maximum wavelength to 28um, common for pRT3
        pRT_wave = pRT_wave[pRT_wave <= 28e-4]
        

    # Create directory if not exist
    if hasattr(conf, 'pRT3_output_dir'):
        pRT3_output_dir = conf.pRT3_output_dir
        assert pathlib.Path(pRT3_output_dir).exists(), f"Output directory {pRT3_output_dir} does not exist"
    else:
        pRT3_output_dir = f'{Data.output_dir}/pRT3/'
        pathlib.Path(pRT3_output_dir).mkdir(parents=True, exist_ok=True)


    # (wave.size, P.size, T.size) -> (P.size, T.size, wave.size)
    sigma = np.moveaxis(sigma, 0, -1)
    
    # Interpolate onto pRT's wavelength-grid
    new_sigma = np.zeros((sigma.shape[0], sigma.shape[1], pRT_wave.size))
    
    for idx_P in range(len(P_grid)):
        for idx_T in range(len(T_grid)):
            new_sigma[idx_P,idx_T] = np.interp(
                pRT_wave, wave_cm, sigma[idx_P,idx_T], left=0.0, right=0.0
                )
            
    # 2) Convert sigma to variable `xsecarr` 
    wavenumbers = 1 / pRT_wave
    wavenumbers = wavenumbers[::-1]
    wave_um_pRT = 1e4 * (1 / wavenumbers) # [cm^-1] -> [um]
    
    # check valid wavelength range
    wave_um_min = max(wave_um.min(), wave_um_pRT.min())
    wave_um_max = min(wave_um.max(), wave_um_pRT.max())
    
    xsecarr = new_sigma[:,:,::-1] # (P, T, wavenumber)
    del sigma
    
    if debug:
        print(f'[convert_to_pRT3_format] wave_um min: {wave_um.min():.2e} um wave_um max: {wave_um.max():.2e} um')
        print(f'[convert_to_pRT3_format] xsecarr shape: {xsecarr.shape}')
    
    
    # Save in pRT3 format following:
    # https://gitlab.com/mauricemolli/petitRADTRANS/-/blob/master/petitRADTRANS/__file_conversion.py

    # examples: 
    # 1) isotopologue_id={'K':39}           --> '39K'
    # 2) isotopologue_id={'C': 12, 'O': 16} --> '12C-16O'
    # 3) isotopologue_id={'H2':1, 'O':18}   --> '1H2-18O'
    # 4) isotopologue_id={'C':12, 'H4':1}   --> '12C-1H4'
    isotopologue_id = getattr(conf, 'isotopologue_id', None)
    assert isotopologue_id is not None, 'isotopologue_id must be defined in input file'

    # species = "".join(list(isotopologue_id.keys())) # this leads to errors: e.g. 'H2O' -> 'HO'
    species = getattr(conf, 'species', 'UNKNOWN')
    # species_isotopologue_name = "-".join([f"{v}{k}" for k,v in isotopologue_id.items()])
    if getattr(conf, 'element_count', None) is not None:
        species_isotopologue_name_list = []
        for key, value in isotopologue_id.items():
            n_str = str(conf.element_count[key]) if conf.element_count[key] > 1 else ''
            species_isotopologue_name_list.append(f"{value}{key}{n_str}")
            
        species_isotopologue_name = "-".join(species_isotopologue_name_list)
    else:  
        species_isotopologue_name = "-".join([f"{v}{k}" for k,v in isotopologue_id.items()])
        
    print(f"[convert_to_pRT3_format] Species isotopologue name: {species_isotopologue_name}")
    
    # mass = Data.atoms_info.loc[species,'mass']
    mass = getattr(conf, 'mass', None)
    assert mass is not None, 'mass must be defined in input file'
    if debug:
        print(f"[convert_to_pRT3_format] Atomic mass of {species_isotopologue_name}: {mass}")
    
    source = getattr(Data, 'database', 'UNKNOWN')
    custom_label = getattr(conf, 'custom_label', '')
    source += f'_{custom_label}' if custom_label != '' else ''
    
    if source == 'UNKNOWN':
        print('WARNING: No database source found, set using "database" attribute. Defaulting to "UNKNOWN"')
        
    resolving_power = kwargs.get('resolving_power', 1e6)
    file_pRT3 = get_opacity_filename(
        resolving_power=resolving_power,
        wavelength_boundaries=[wave_um_min, wave_um_max],
        species_isotopologue_name=species_isotopologue_name,
        source=source,
    )

    # output_dir = pathlib.Path("/net/lem/data2/pRT3/input_data/opacities/lines/line_by_line/") / species / species_isotopologue_name
    # assert output_dir.exists(), f"Output directory {output_dir} does not exist"
    out_dir_species = pathlib.Path(pRT3_output_dir) / species / species_isotopologue_name
    out_dir_species.mkdir(parents=True, exist_ok=True)
    
    hdf5_opacity_file = out_dir_species / f'{file_pRT3}.xsec.petitRADTRANS.h5'
    if debug:
        print(f'[convert_to_pRT3_format] Saving to file: {hdf5_opacity_file}...')

    print(hdf5_opacity_file)
    doi = kwargs.get('doi', 'None')
    contributor = kwargs.get('contributor', 'LEM') # default to LEM (Leiden Exoplanet Machine)
    description = kwargs.get('description', 'Converted from `cs_package` format to `pRT3` format')

    write_line_by_line(
        hdf5_opacity_file, 
        doi, 
        wavenumbers, 
        xsecarr, 
        mass, 
        species, 
        np.unique(P_grid), 
        np.unique(T_grid), 
        wavelengths=None, 
        contributor=contributor, 
        description=description
        )
    
# adapted from https://gitlab.com/mauricemolli/petitRADTRANS/-/blob/master/petitRADTRANS/__file_conversion.py
def write_line_by_line(file, doi, wavenumbers, opacities, mol_mass, species,
                       opacities_pressures, opacities_temperatures, wavelengths=None,
                       contributor=None, description=None,
                       pRT_version="3.0.7"):
    import datetime
    
    if wavelengths is None:
        wavelengths = np.array([1 / wavenumbers[0], 1 / wavenumbers[-1]])

    with h5py.File(file, "w") as fh5:
        dataset = fh5.create_dataset(
            name='DOI',
            shape=(1,),
            data=doi
        )
        dataset.attrs['long_name'] = 'Data object identifier linked to the data'
        dataset.attrs['contributor'] = str(contributor)
        dataset.attrs['additional_description'] = str(description)

        dataset = fh5.create_dataset(
            name='Date_ID',
            shape=(1,),
            data=f'petitRADTRANS-v{pRT_version}_{datetime.datetime.now(datetime.timezone.utc).isoformat()}'
        )
        dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                     'at which the table has been created, ' \
                                     'along with the version of petitRADTRANS'

        dataset = fh5.create_dataset(
            name='bin_edges',
            data=wavenumbers
        )
        dataset.attrs['long_name'] = 'Wavenumber grid'
        dataset.attrs['units'] = 'cm^-1'

        dataset = fh5.create_dataset(
            name='xsecarr',
            data=opacities
        )
        dataset.attrs['long_name'] = 'Table of the cross-sections with axes (pressure, temperature, wavenumber)'
        dataset.attrs['units'] = 'cm^2/molecule'

        dataset = fh5.create_dataset(
            name='mol_mass',
            shape=(1,),
            data=float(mol_mass)
        )
        dataset.attrs['long_name'] = 'Mass of the species'
        dataset.attrs['units'] = 'AMU'

        dataset = fh5.create_dataset(
            name='mol_name',
            shape=(1,),
            data=species.split('_', 1)[0]
        )
        dataset.attrs['long_name'] = 'Name of the species described'

        dataset = fh5.create_dataset(
            name='p',
            data=opacities_pressures
        )
        dataset.attrs['long_name'] = 'Pressure grid'
        dataset.attrs['units'] = 'bar'

        dataset = fh5.create_dataset(
            name='t',
            data=opacities_temperatures
        )
        dataset.attrs['long_name'] = 'Temperature grid'
        dataset.attrs['units'] = 'K'

        dataset = fh5.create_dataset(
            name='temperature_grid_type',
            shape=(1,),
            data='regular'
        )
        dataset.attrs['long_name'] = 'Whether the temperature grid is "regular" ' \
                                     '(same temperatures for all pressures) or "pressure-dependent"'

        dataset = fh5.create_dataset(
            name='wlrange',
            data=np.array([wavelengths.min(), wavelengths.max()]) * 1e4  # cm to um
        )
        dataset.attrs['long_name'] = 'Wavelength range covered'
        dataset.attrs['units'] = 'µm'

        dataset = fh5.create_dataset(
            name='wnrange',
            data=np.array([wavenumbers.min(), wavenumbers.max()])
        )
        dataset.attrs['long_name'] = 'Wavenumber range covered'
        dataset.attrs['units'] = 'cm^-1'
        
    print(f"Saved to {file}")
    return None


def get_opacity_filename(resolving_power, wavelength_boundaries, species_isotopologue_name,
                         source):
    if resolving_power < 1e6:
        resolving_power = f"{resolving_power:.0f}"
    else:
        decimals = np.mod(resolving_power / 10 ** np.floor(np.log10(resolving_power)), 1)

        if decimals >= 1e-3:
            resolving_power = f"{resolving_power:.3e}"
        else:
            resolving_power = f"{resolving_power:.0e}"

    spectral_info = (f"R{resolving_power}_"
                     f"{wavelength_boundaries[0]:.1f}-{wavelength_boundaries[1]:.1f}mu")

    return join_species_all_info(
        name=species_isotopologue_name,
        source=source,
        spectral_info=spectral_info
    )
    
def join_species_all_info(name, natural_abundance='', charge='', cloud_info='', source='', spectral_info=''):
    if natural_abundance != '':
        name += '-' + natural_abundance

    name += charge + cloud_info

    if source != '':
        name += '__' + source

    if spectral_info != '':
        name += '.' + spectral_info

    return name
