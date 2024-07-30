import numpy as np

database = 'ExoMol'

# Instructions to download from ExoMol database
url_def_json = 'https://www.exomol.com/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL.json'
url_broad = [
    'https://www.exomol.com/db/H2O/1H2-16O/1H2-16O__H2.broad', 
    'https://www.exomol.com/db/H2O/1H2-16O/1H2-16O__He.broad'
]
out_dir = '/net/lem/data2/regt/cs_package/linelists/h2o/exomol/'

files = dict(
    partition_function = '/net/lem/data2/regt/cs_package/linelists/h2o/exomol/1H2-16O__POKAZATEL.pf', 
    H2_broadening = '/net/lem/data2/regt/cs_package/linelists/h2o/exomol/1H2-16O__H2.broad', 
    He_broadening = '/net/lem/data2/regt/cs_package/linelists/h2o/exomol/1H2-16O__He.broad', 
    transitions = [
        '/net/lem/data2/regt/cs_package/linelists/h2o/exomol/1H2-16O__POKAZATEL__{:05d}-{:05d}.trans.bz2'.format(nu_min, nu_min+100) \
        for nu_min in np.arange(0, 41200, 100)
        ], 
    states = '/net/lem/data2/regt/cs_package/linelists/h2o/exomol/1H2-16O__POKAZATEL.states.bz2', 
    
    tmp_output = '/net/lem/data2/regt/cs_package/cross_sec_outputs/tmp/h2o_cross{}.hdf5', 
    final_output = '/net/lem/data2/regt/cs_package/cross_sec_outputs/h2o/h2o_exomol.hdf5', 
)

pRT = dict(
    out_dir = '/net/lem/data2/regt/cs_package/cross_sec_outputs/h2o/h2o_exomol_pRT2/', 
    wave = '/net/lem/data1/regt/pRT_opacities/data/wlen_petitRADTRANS.dat'
)

P_grid = np.logspace(-5,2,8) # [bar]
T_grid = np.array(
    [300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000], dtype=np.float64
    ) # [K]

mass = 18.010565

wave_min = 1.0/3.0; wave_max = 50.0 # um
#wave_min = 2.0; wave_max = 3.0
delta_nu = 0.01

local_cutoff  = 0.35
global_cutoff = 1e-45

# gamma_V [cm^-1], P [bar]
#wing_cutoff = lambda gamma_V, _: 100*gamma_V
wing_cutoff = lambda _, P: 25 if P<=200 else 100 # Gharib-Nezhad et al. (2024)