import numpy as np

database = 'ExoMol'

# Instructions to download from ExoMol database
url_def_json = 'https://www.exomol.com/db/AlH/27Al-1H/AloHa/27Al-1H__AloHa.json'
url_broad = [
    'https://www.exomol.com/db/AlH/27Al-1H/27Al-1H__H2.broad', 
    'https://www.exomol.com/db/AlH/27Al-1H/27Al-1H__He.broad'
]
input_dir = '/net/lem/data2/regt/pRT_opacities/input_data/ExoMol/AlH/'

# Output-directory
cross_sec_outputs = '/net/lem/data2/regt/pRT_opacities/cross_sec_outputs/'

files = dict(
    partition_function = f'{input_dir}/27Al-1H__AloHa.pf', 
    H2_broadening      = f'{input_dir}/27Al-1H__H2.broad', 
    He_broadening      = f'{input_dir}/27Al-1H__He.broad', 
    transitions        = [f'{input_dir}/27Al-1H__AloHa.trans.bz2'], 
    states             = f'{input_dir}/27Al-1H__AloHa.states.bz2', 
    
    tmp_output   = f'{cross_sec_outputs}'+'/AlH/tmp/AlH_cross{}.hdf5', 
    final_output = f'{cross_sec_outputs}/AlH/AlH.hdf5', 
)

pRT = dict(
    out_dir         = f'{cross_sec_outputs}/AlH/AlH_pRT2/', 
    wave            = '/net/lem/data1/regt/pRT_opacities/data/wlen_petitRADTRANS.dat', 
    make_short_file = '/net/lem/data2/regt/pRT_opacities/input_data/make_short.f90', 
)

P_grid = np.logspace(-5,2,8) # [bar]
T_grid = np.array(
    [300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000], dtype=np.float64
    ) # [K]

#P_grid = 10**np.array([-6,3], dtype=np.float64)
#T_grid = np.array([300,5000])

#P_grid = np.array([0.1,1.,10.]) # [bar]
#T_grid = np.array([500,1000,2000], dtype=np.float64) # [K]

mass = 27.98948 # (in .json file)

wave_min = 1.0/3.0; wave_max = 50.0 # [um]
delta_nu = 0.01 # [cm^-1]

# Line-strength cutoffs
local_cutoff  = 0.35
global_cutoff = 1e-45

# gamma_V [cm^-1], P [bar]
wing_cutoff = lambda gamma_V, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024)