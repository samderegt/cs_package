import numpy as np

database = 'VALD'

species = 'K'
input_dir = './input_data/VALD/K/'

# Output-directory
cross_sec_outputs = './cross_sec_outputs/K/'

files = dict(
    # https://www.astro.uu.se/valdwiki
    # 'Extract element' | 'short format' | 'FTP'
    # !! make sure to get units in vacuum (cm^-1) !!
    transitions = f'{input_dir}/VALD_transitions.txt',

    # https://physics.nist.gov/PhysRefData/ASD/levels_form.html 
    # !! Request .csv (units: cm^-1) with degeneracy g !!
    states = f'{input_dir}/NIST_levels_tab_delimited.tsv', 
    
    tmp_output   = f'{cross_sec_outputs}/tmp/{species}'+'_cross{}.hdf5', 
    final_output = f'{cross_sec_outputs}/{species}.hdf5', 
)

pRT = dict(
    out_dir         = f'{cross_sec_outputs}/{species}_pRT2/', 
    wave            = './input_data/wlen_petitRADTRANS.dat', 
    make_short_file = './input_data/make_short.f90', 
)

P_grid = np.logspace(-5,3,9) # [bar]
T_grid = np.array([
    81.14113604736988, 
    109.60677358237457, 
    148.05862230132453, 
    200., 
    270.163273706, 
    364.940972297, 
    492.968238926, 
    665.909566306, 
    899.521542126, 
    1215.08842295, 
    1641.36133093, 
    2217.17775249, 
    2995., 
    ]) # [K]

mass = 39.0983

# !! If following parameters are given, use vdW-prescription from Schweitzer+ (1996) !!
# !! Only for alkali's !!
E_ion = 35009.8140  # [cm^-1]  https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
Z = 1 # Electric charge (Lacy & Burrows 2023)

broadening = dict(
    H2={'VMR':0.85, 'mass':2.01568, 'alpha':0.806e-24}, 
    He={'VMR':0.15, 'mass':4.002602, 'alpha':0.204956e-24}

    # (Kurucz & Furenlid 1979)
    #H2={'VMR':0.85, 'C':0.85}, 
    #He={'VMR':0.15, 'C':0.42}, 
    )

# !! Ignores certain lines in 'transitions'-file !!
#nu_0_to_ignore = [16956.1701, 16973.3661] # Na I optical resonance doublet

wave_min = 1.0/3.0; wave_max = 50.0 # [um]
delta_nu = 0.01 # [cm^-1]

# Line-strength cutoffs
local_cutoff  = None
global_cutoff = None

# gamma_V [cm^-1], P [bar]
#wing_cutoff = lambda gamma_V, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024)
wing_cutoff = lambda gamma_V, P: 1000 # !! Might want different wing-cutoff for atoms !!