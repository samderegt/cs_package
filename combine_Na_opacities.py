import numpy as np

temperatures_fine = np.array([
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
])
pressures = 10**np.arange(-6,3+1e-6,1)

wave_pRT_grid = np.loadtxt('./data/wlen_petitRADTRANS.dat').T

#path_base = '/net/lem/data2/regt/Na_I_opacities_recomputed_pRT_grid/'
path_base = '/net/lem/data2/regt/Na_I_recomputed_3/'

for P in pressures:
    for T in temperatures_fine:

        print('sigma_{:.0f}.K_{:.6f}bar.dat'.format(T,P))

        # Add the other lines and save cross-sections
        path_other_lines = path_base + \
            'wo_doublet/sigma_{:.0f}.K_{:.6f}bar.dat'.format(T, P)
        _, sigma_other_lines = np.loadtxt(path_other_lines).T

        path_doublet = path_base + \
            'opacities_doublet/D1D2_T{:.0f}_P{:.6f}.dat'.format(T, P)
        sigma_doublet = np.loadtxt(path_doublet).T

        # Combine doublet and other lines
        sigma_tot = sigma_other_lines + sigma_doublet

        # Save wavelength and cross-section
        np.savetxt(
            path_base + 'sigma_{:.0f}.K_{:.6f}bar.dat'.format(T,P), 
            np.column_stack((wave_pRT_grid, sigma_tot))
            )