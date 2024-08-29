import numpy as np

database = 'HITEMP'

# Instructions to download from HITRAN/HITEMP database
urls = [
    # Transitions (see https://hitran.org/hitemp/)
    'https://hitran.org/hitemp/data/bzip2format/05_HITEMP2019.par.bz2', 

    # Partition function (see https://hitran.org/docs/iso-meta/)
    'https://hitran.org/data/Q/q26.txt', 

    # Broadening parameters (from ExoMol)
    'https://www.exomol.com/db/CO/12C-16O/12C-16O__H2.broad', 
    'https://www.exomol.com/db/CO/12C-16O/12C-16O__He.broad', 
]
input_dir = './input_data/HITEMP/12CO/'

# Output-directory
cross_sec_outputs = './cross_sec_outputs/'

files = dict(
    partition_function = f'{input_dir}/q26.txt', 
    transitions = f'{input_dir}/05_HITEMP2019.par.bz2', 
    
    tmp_output   = f'{cross_sec_outputs}'+'/12CO/tmp/12CO_cross{}.hdf5', 
    final_output = f'{cross_sec_outputs}/12CO/12CO.hdf5', 
)

broadening = dict(
    H2 = dict(
        VMR=0.85, 
        file=f'{input_dir}/12C-16O__H2.broad', # read from file
        #gamma=0.1, n=0.5,                       # manually provide parameters
        ), 
    He = dict(
        VMR=0.15, 
        file=f'{input_dir}/12C-16O__He.broad', # read from file
        #gamma=0.1, n=0.5                        # manually provide parameters
        ), 
)

pRT = dict(
    out_dir         = f'{cross_sec_outputs}/12CO/12CO_pRT2/', 
    wave            = './input_data/wlen_petitRADTRANS.dat', 
    make_short_file = './input_data/make_short.f90', 
)

P_grid = np.logspace(-5,2,8) # [bar]
T_grid = np.array(
    #[300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000], dtype=np.float64
    [500], dtype=np.float64
    ) # [K]


mass = 28.01
isotope_idx = 1 # !! HITEMP specific !! .par-file includes all isotopologues

wave_min = 1.0/3.0; wave_max = 50.0 # [um]
delta_nu = 0.01 # [cm^-1]

# Line-strength cutoffs
local_cutoff  = 0.35
global_cutoff = None

# gamma_V [cm^-1], P [bar]
wing_cutoff = lambda gamma_V, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024)