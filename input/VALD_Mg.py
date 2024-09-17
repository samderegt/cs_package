import numpy as np

database = 'Kurucz'

species = 'Mg'
isotopologue_id = dict(Mg=12)
input_dir = f'./input_data/VALD/{species}/'

# Output-directory
cross_sec_outputs = f'./cross_sec_outputs/{species}/'

files = dict(
    # https://www.astro.uu.se/valdwiki
    # 'Extract element' | 'short format' | 'FTP'
    # !! make sure to get units in vacuum (cm^-1) !!
    #transitions = f'{input_dir}/VALD_transitions.txt',
    transitions = f'{input_dir}/Kurucz_transitions.txt',

    # https://physics.nist.gov/PhysRefData/ASD/levels_form.html 
    # !! Request .csv (units: cm^-1) with degeneracy g !!
    states = f'{input_dir}/NIST_levels_tab_delimited.tsv', 
    
    tmp_output_dir   = f'{cross_sec_outputs}/tmp/{species}'+'_cross{}.hdf5', 
    final_output = f'{cross_sec_outputs}/{species}.hdf5', 
)

pRT = dict(
    out_dir         = f'{cross_sec_outputs}/{species}_pRT2/', 
    wave            = './input_data/wlen_petitRADTRANS.dat', 
    make_short_file = './input_data/make_short.f90', 
)

P_grid = np.logspace(-5,2,8) # [bar]
T_grid = np.array([3000.]) # [K]

mass = 24.305

broadening = dict(
    H2={'VMR':0.85, 'mass':2.01568, 'alpha':0.806e-24}, 
    He={'VMR':0.15, 'mass':4.002602, 'alpha':0.204956e-24}

    # (Kurucz & Furenlid 1979)
    #H2={'VMR':0.85, 'C':0.85}, 
    #He={'VMR':0.15, 'C':0.42}, 
    )

wave_min = 1.0/3.0; wave_max = 50.0 # [um]
delta_nu = 0.01 # [cm^-1]
adaptive_nu_grid = False

# Line-strength cutoffs
local_cutoff  = None
global_cutoff = None

# gamma_V [cm^-1], P [bar]
#wing_cutoff = lambda gamma_V, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024)
wing_cutoff = lambda gamma_V, P: 1000 # !! Might want different wing-cutoff for atoms !!