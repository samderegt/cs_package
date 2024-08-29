import numpy as np

database = 'ExoMol'

# Instructions to download from ExoMol database
url_def_json = 'https://www.exomol.com/db/NaH/23Na-1H/Rivlin/23Na-1H__Rivlin.json'
url_broad = [
    'https://www.exomol.com/db/NaH/23Na-1H/23Na-1H__H2.broad', 
    'https://www.exomol.com/db/NaH/23Na-1H/23Na-1H__He.broad'
]
input_dir = './input_data/ExoMol/NaH/'

# Output-directory
cross_sec_outputs = './cross_sec_outputs/'

files = dict(
    partition_function = f'{input_dir}/23Na-1H__Rivlin.pf', 
    transitions        = [f'{input_dir}/23Na-1H__Rivlin.trans.bz2'], 
    states             = f'{input_dir}/23Na-1H__Rivlin.states.bz2', 

    tmp_output   = f'{cross_sec_outputs}'+'/NaH/tmp/NaH_cross{}.hdf5', 
    final_output = f'{cross_sec_outputs}/NaH/NaH.hdf5', 
)

broadening = dict(
    H2 = dict(
        VMR=0.85, 
        file=f'{input_dir}/23Na-1H__H2.broad', # read from file
        #gamma=0.1, n=0.5,                      # manually provide parameters
        ), 
    He = dict(
        VMR=0.15, 
        file=f'{input_dir}/23Na-1H__H2.broad', # read from file
        #gamma=0.1, n=0.5                       # manually provide parameters
        ), 
)

pRT = dict(
    out_dir         = f'{cross_sec_outputs}/NaH/NaH_pRT2/', 
    wave            = './input_data/wlen_petitRADTRANS.dat', 
    make_short_file = './input_data/make_short.f90', 
)

P_grid = np.logspace(-5,2,8) # [bar]
T_grid = np.array(
    #[300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000], dtype=np.float64
    [500], dtype=np.float64
    ) # [K]


mass = 23.997594 # (in .json file)

wave_min = 1.0/3.0; wave_max = 50.0 # [um]
delta_nu = 0.01 # [cm^-1]

# Line-strength cutoffs
local_cutoff  = 0.35
global_cutoff = 1e-45

# gamma_V [cm^-1], P [bar]
wing_cutoff = lambda gamma_V, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024)