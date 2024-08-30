import numpy as np

database = 'ExoMol'

# Instructions to download from ExoMol database
url_def_json = 'https://www.exomol.com/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL.json'
url_broad = [
    'https://www.exomol.com/db/H2O/1H2-16O/1H2-16O__H2.broad', 
    'https://www.exomol.com/db/H2O/1H2-16O/1H2-16O__He.broad'
]
input_dir = './input_data/ExoMol/H2O/'

# Output-directory
cross_sec_outputs = './cross_sec_outputs/'

files = dict(
    partition_function = f'{input_dir}/1H2-16O__POKAZATEL.pf', 
    transitions = [
        '{}/1H2-16O__POKAZATEL__{:05d}-{:05d}.trans.bz2'.format(input_dir, nu_min, nu_min+100) \
        for nu_min in np.arange(0, 41200, 100)
        ], 
    states = f'{input_dir}/1H2-16O__POKAZATEL.states.bz2', 

    tmp_output   = f'{cross_sec_outputs}'+'/H2O/tmp/H2O_cross{}.hdf5', 
    final_output = f'{cross_sec_outputs}/H2O/H2O.hdf5', 
)

broadening = dict(
    H2 = dict(
        VMR=0.85, file=f'{input_dir}/1H2-16O__H2.broad', # read from file
        ), 
    He = dict(
        VMR=0.15, file=f'{input_dir}/1H2-16O__He.broad', # read from file
        ), 
)

pRT = dict(
    out_dir         = f'{cross_sec_outputs}/H2O/H2O_pRT2/', 
    wave            = './input_data/wlen_petitRADTRANS.dat', 
    make_short_file = './input_data/make_short.f90', 
)

P_grid = np.logspace(-5,2,8) # [bar]
T_grid = np.array(
    [300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000], dtype=np.float64
    ) # [K]

mass = 18.010565 # (in .json file)

wave_min = 1.0/3.0; wave_max = 50.0 # [um]
delta_nu = 0.01 # [cm^-1]

local_cutoff  = 0.35
global_cutoff = 1e-45

# gamma_V [cm^-1], P [bar]
wing_cutoff = lambda _, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024)