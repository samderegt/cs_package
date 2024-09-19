import numpy as np

database = 'ExoMol'
species  = 'H2O'
mass = 18.010565 # (in .json file)

# Instructions to download from ExoMol database
url_def_json = 'https://www.exomol.com/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL.def.json'
url_broad = [
    'https://www.exomol.com/db/H2O/1H2-16O/1H2-16O__H2.broad', 
    'https://www.exomol.com/db/H2O/1H2-16O/1H2-16O__He.broad'
]

# Input/output-directories
input_dir  = f'./input_data/{database}/{species}/'
output_dir = f'./cross_sec_outputs/{species}/'

files = dict(
    transitions = [
        '{}/1H2-16O__POKAZATEL__{:05d}-{:05d}.trans.bz2'.format(input_dir, nu_min, nu_min+100) \
        for nu_min in np.arange(0, 41200, 100)
    ], 
    states = f'{input_dir}/1H2-16O__POKAZATEL.states.bz2', 

    partition_function = f'{input_dir}/1H2-16O__POKAZATEL.pf', 
)

# Pressure-broadening information
broadening = dict(
    H2 = dict(
        VMR=0.85, file=f'{input_dir}/1H2-16O__H2.broad', # read from file
        ), 
    He = dict(
        VMR=0.15, file=f'{input_dir}/1H2-16O__He.broad', # read from file
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