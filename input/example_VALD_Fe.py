import numpy as np

database = 'VALD'
species  = 'Fe'
isotopologue_id = {'Fe':56}

# Input/output-directories
input_dir  = f'./input_data/{database}/{species}/'
output_dir = f'./cross_sec_outputs/{species}/'

files = dict(
    transitions = f'{input_dir}/{database}_transitions.txt', # VALD transitions
    states = f'{input_dir}/NIST_levels_tab_delimited.tsv',   # NIST energy levels

    #pRT_wave       = './input_data/wlen_petitRADTRANS.dat', # DEFAULT
    #pRT_make_short = './input_data/make_short.f90',         # DEFAULT
)

# Pressure-broadening information
#broadening = dict(
#    H2={'VMR':0.85, 'mass':2.01568, 'alpha':0.806e-24},    # DEFAULT for database='Kurucz'
#    He={'VMR':0.15, 'mass':4.002602, 'alpha':0.204956e-24} # DEFAULT for database='Kurucz'
#    #H2={'VMR':0.85, 'C':0.85}, He={'VMR':0.15, 'C':0.42}, # (Kurucz & Furenlid 1979)
#    )

P_grid = np.logspace(-5,2,8)        # [bar]   # can be given in cmd, one point at a time
T_grid = np.array([1000,2000,3000]) # [K]     # can be given in cmd, one point at a time

#wave_min = 1.0/3.0; wave_max = 50.0 # [um]
wave_min = 1.0; wave_max = 1.5 # [um]
delta_nu = 0.01 # [cm^-1]

# Switch to sparser wavenumber grid for high broadening?
#adaptive_nu_grid = False # DEFAULT

# Line-strength cutoffs
#local_cutoff  = None # DEFAULT
#global_cutoff = None # DEFAULT

# gamma_V [cm^-1], P [bar]
#wing_cutoff = lambda _, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024) DEFAULT
