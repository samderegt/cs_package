import numpy as np

database = 'VALD'

files = dict(
    transitions = '/net/lem/data2/regt/pRT_opacities/input_data/VALD/Fe/Fe_I_transitions.txt', 
    states = '/net/lem/data2/regt/pRT_opacities/input_data/VALD/Fe/energy1.tsv', 
    
    tmp_output = '/net/lem/data2/regt/pRT_opacities/cross_sec_outputs/tmp/fe_cross{}.hdf5', 
    final_output = '/net/lem/data2/regt/pRT_opacities/cross_sec_outputs/fe/fe_vald.hdf5', 
)

pRT = dict(
    out_dir = '/net/lem/data2/regt/pRT_opacities/cross_sec_outputs/fe/fe_vald_pRT2/', 
    wave = '/net/lem/data1/regt/pRT_opacities/data/wlen_petitRADTRANS.dat', 
    make_short_file = '/net/lem/data2/regt/petitRADTRANS_opa_source/make_short.f90', 
)

#P_grid = np.logspace(-5,2,8) # [bar]
#T_grid = np.array(
#    [300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000], dtype=np.float64
#    ) # [K]

P_grid = np.logspace(-1,2,4) # [bar]
T_grid = np.array(
    [500,1000,2000,3000,4000,5000], dtype=np.float64
    ) # [K]

mass = 55.845

#wave_min = 1.0/3.0; wave_max = 50.0 # um
wave_min = 0.3; wave_max = 50.0 # um
#wave_min = 2.0; wave_max = 3.0
delta_nu = 0.01

local_cutoff  = 0.35
global_cutoff = 1e-45

# gamma_V [cm^-1], P [bar]
#wing_cutoff = lambda gamma_V, _: 25*gamma_V # Gandhi et al. (2020)
#wing_cutoff = lambda _, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024)
wing_cutoff = lambda gamma_V, P: 100 # Is this okay?