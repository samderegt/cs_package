import numpy as np

database = 'ExoMol'
species  = 'CaH'
mass = 40.970416 # (in .json file)
isotopologue_id = {'Ca':40, 'H':1}

# Instructions to download from ExoMol database
url_def_json = 'https://www.exomol.com/db/CaH/40Ca-1H/XAB/40Ca-1H__XAB.def.json'
url_broad = [
    'https://www.exomol.com/db/CaH/40Ca-1H/40Ca-1H__H2.broad', 
    'https://www.exomol.com/db/CaH/40Ca-1H/40Ca-1H__He.broad'
]

# Output-directory
input_dir  = f'./input_data/{database}/{species}/'
output_dir = f'./cross_sec_outputs/{species}/'

files = dict(
    transitions = f'{input_dir}/40Ca-1H__XAB.trans.bz2', 
    states      = f'{input_dir}/40Ca-1H__XAB.states.bz2', 

    partition_function = f'{input_dir}/40Ca-1H__XAB.pf', 
)

from broaden import Gharib_Nezhad_ea_2021 as B
broadening = dict(
    H2 = dict(
        VMR=0.85, gamma=B('CaH').gamma_H2, n=0.5, # use 4th-order Pade equation (input=J_lower)
        ), 
    He = dict(
        VMR=0.15, gamma=B('CaH').gamma_He, n=0.5, # use 4th-order Pade equation (input=J_lower)
        ), 
)

P_grid = np.logspace(-5,2,8) # [bar]   # can be given in cmd, one point at a time
T_grid = np.array([1000,])   # [K]     # can be given in cmd, one point at a time

#wave_min = 1.0/3.0; wave_max = 50.0 # [um]
wave_min = 1.0; wave_max = 1.5 # [um]
delta_nu = 0.01 # [cm^-1]

# Switch to sparser wavenumber grid for high broadening?
adaptive_nu_grid = True

# Line-strength cutoffs
local_cutoff  = 0.35
global_cutoff = 1e-45

# gamma_V [cm^-1], P [bar]
#wing_cutoff = lambda _, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024) DEFAULT