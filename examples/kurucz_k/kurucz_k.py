# Basic information on database and species
database = 'kurucz' # Can be ['exomol', 'hitran', 'hitemp', 'kurucz']
species  = 'k'      # Species name
# mass = 39.0983     # Will be read from atoms_info.csv

N_CPUs = 4 # Number of CPUs for parallelisation

# Input/output-directories
input_data_dir  = './examples/kurucz_k/input_data/'
output_data_dir = './examples/kurucz_k/'

# Downloading can be done automatically from the Kurucz database
# urls = []

# Input-data files
files = dict(
    transitions = f'{input_data_dir}/gf1900.pos',
    states      = f'{input_data_dir}/NIST_states.tsv',
)

import numpy as np
# Pressure and temperature grids
P_grid = np.logspace(-3, 2, 6) # [bar]
T_grid = np.array([1000,2000]) # [K]

# Wavenumber grid
wave_min = 0.3; wave_max = 28.0 # [um]
delta_nu = 0.01 # [cm^-1]

# Power-law impact-width/shift parameters gamma=A*T^b*n_perturber/n_reference
# impact_width_info = dict(
#     H2 = dict(VMR=0.85, A=[], b=[], nu_0=[], n_reference=1e20), 
#     He = dict(VMR=0.15, A=[], b=[], nu_0=[], n_reference=1e20),
# )
# impact_shift_info = dict(
#     H2 = dict(VMR=0.85, A=[], b=[], nu_0=[], n_reference=1e20), 
#     He = dict(VMR=0.15, A=[], b=[], nu_0=[], n_reference=1e20),
# )

# Ignore quantum transitions to include custom line profiles outside of pyROX
# ignore_transitions = [
#     (0.5, '4s 2S', 1.5, '4p 2P'), # J_1, label_1, J_2, label_2
#     (0.5, '4s 2S', 0.5, '4p 2P'), 
#     ]

# Pressure-broadening information
# perturber_info = {} # K is an alkali, defaults to Schweitzer et al. (1996) description

# Function with arguments gamma_V [cm^-1], and P [bar]
wing_cutoff = lambda *_: 1500

# Metadata to be stored in pRT3's .h5 file
pRT3_metadata = dict(
    DOI = 'http://kurucz.harvard.edu/atoms/', # DOI of the data
    mol_name = 'K',                           # Using the right capitalisation
    isotopologue_id = {'K':19},               # Atomic number of each element
)