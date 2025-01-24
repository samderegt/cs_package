import numpy as np

database = 'HITEMP'
species  = '12CO'
mass = 28.01 
isotope_idx = 1 # !! HITEMP specific !! .par-file includes all isotopologues
isotope_abundance = 9.86544e-1
isotopologue_id = {'C':12, 'O':16}

# Instructions to download from HITRAN/HITEMP database
urls = [
    # Transitions (see https://hitran.org/hitemp/)
    #'https://hitran.org/hitemp/data/bzip2format/05_HITEMP2019.par.bz2', # DEPRECATED (requires login)

    # Partition function (see https://hitran.org/docs/iso-meta/)
    'https://hitran.org/data/Q/q26.txt', 

    # Broadening parameters (from ExoMol)
    'https://www.exomol.com/db/CO/12C-16O/12C-16O__H2.broad', 
    'https://www.exomol.com/db/CO/12C-16O/12C-16O__He.broad', 
]

# Input/output-directories
input_dir  = f'./input_data/{database}/{species}/'
output_dir = f'./cross_sec_outputs/{species}/'

files = dict(
    transitions = f'{input_dir}/05_HITEMP2019.par.bz2', 

    partition_function = f'{input_dir}/q26.txt', 
)

broadening = dict(
    H2 = dict(
        VMR=0.85, file=f'{input_dir}/12C-16O__H2.broad', # read from file
        ), 
    He = dict(
        VMR=0.15, file=f'{input_dir}/12C-16O__He.broad', # read from file
        ), 
)

P_grid = np.logspace(-5,2,8) # [bar]   # can be given in cmd, one point at a time
T_grid = np.array([1000,])   # [K]     # can be given in cmd, one point at a time

wave_min = 1.0/3.0; wave_max = 50.0 # [um]
delta_nu = 0.01 # [cm^-1]

# Switch to sparser wavenumber grid for high broadening?
adaptive_nu_grid = True

# Line-strength cutoffs
local_cutoff  = 0.35
global_cutoff = 1e-45

# gamma_V [cm^-1], P [bar]
#wing_cutoff = lambda _, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024) DEFAULT