import numpy as np

database = 'ExoMol'

input_dir = '/net/lem/data2/regt/pRT_opacities/input_data/ExoMol/15NH3/'

# Output-directory
cross_sec_outputs = '/net/lem/data2/regt/pRT_opacities/cross_sec_outputs/'

files = dict(
    partition_function = f'{input_dir}/15N-1H3__CoYute-15.pf', 
    H2_broadening      = f'{input_dir}/14N-1H3__H2.broad', 
    He_broadening      = f'{input_dir}/14N-1H3__He.broad', 

    transitions = [
        '{}/15N-1H3__CoYute-15__{:05d}-{:05d}.trans.bz2'.format(input_dir, nu_min, nu_min+1000) \
        for nu_min in np.arange(0, 10000, 1000)
        ], 
    states = f'{input_dir}/15N-1H3__CoYute-15.states.bz2', 
    
    tmp_output   = f'{cross_sec_outputs}'+'/15NH3/tmp/15NH3_cross{}.hdf5', 
    final_output = f'{cross_sec_outputs}/15NH3/15NH3.hdf5', 
)

pRT = dict(
    out_dir         = f'{cross_sec_outputs}/15NH3/15NH3_pRT2/', 
    wave            = '/net/lem/data1/regt/pRT_opacities/data/wlen_petitRADTRANS.dat', 
    make_short_file = '/net/lem/data2/regt/pRT_opacities/input_data/make_short.f90', 
)

#P_grid = np.logspace(-5,2,8) # [bar]
#T_grid = np.array(
#    [300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000], dtype=np.float64
#    ) # [K]

P_grid = np.logspace(-5,2,8) # [bar]
T_grid = np.array(
    [300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000], dtype=np.float64
    ) # [K]

mass = 18.023584 # (in .json file)

wave_min = 1.0/3.0; wave_max = 50.0 # [um]
delta_nu = 0.01 # [cm^-1]

local_cutoff  = 0.35
global_cutoff = 1e-45

# gamma_V [cm^-1], P [bar]
wing_cutoff = lambda _, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024)