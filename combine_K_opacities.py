import numpy as np
import matplotlib.pyplot as plt

from scipy.special import wofz as Faddeeva
from scipy.interpolate import interp1d

import glob

e      = 4.8032e-10    # Electron charge, cm^3/2 g^1/2 s^-1
m_e    = 9.1094e-28    # Electron mass, g
c2     = 1.4387769     # cm K
k_B    = 1.3807e-16    # cm^2 g s^-2 K^-1
c      = 2.99792458e10 # cm s^-1
N_A    = 6.02214076e23 # mol^-1
mass_H = 1.6735575e-24 # g
amu    = 1.66054e-24   # g

def nearwing_Lorentz_junction(
        sigma_lorentz, nu_lorentz, sigma_nearwing, nu_nearwing, nu_0, max_nu_offset=200
        ):

    # Left- and right-side of Lorent profile
    mask_1 = (nu_lorentz-nu_0 > -max_nu_offset) & (nu_lorentz-nu_0 < 0)
    mask_2 = (nu_lorentz-nu_0 > 0) & (nu_lorentz-nu_0 < max_nu_offset)

    # Separation (in log) between the 2 components
    #diff = 10**np.interp(nu_lorentz, nu_nearwing, np.log10(sigma_nearwing)) - sigma_lorentz
    diff = np.interp(nu_lorentz, nu_nearwing, np.log10(sigma_nearwing)) - np.log10(sigma_lorentz)

    # Closest opacity on either side
    idx_1 = np.argmin(np.abs(diff[mask_1]))
    nu_lorentz_1 = nu_lorentz[mask_1][idx_1]

    idx_2 = np.argmin(np.abs(diff[mask_2]))
    nu_lorentz_2 = nu_lorentz[mask_2][idx_2]

    # Combine the two profiles
    mask_lorentz  = (nu_lorentz >= nu_lorentz_1) & (nu_lorentz <= nu_lorentz_2)
    mask_nearwing = (nu_nearwing < nu_lorentz_1) | (nu_nearwing > nu_lorentz_2)
    
    sigma_combined = np.concatenate(
        (sigma_nearwing[mask_nearwing], sigma_lorentz[mask_lorentz]), 
        )
    nu_combined = np.concatenate(
        (nu_nearwing[mask_nearwing], nu_lorentz[mask_lorentz]), axis=0
        )
    # Sort by the wavelength
    sigma_combined = sigma_combined[np.argsort(nu_combined)]
    nu_combined    = nu_combined[np.argsort(nu_combined)]

    return sigma_combined, nu_combined

def redwing_junction(sigma_redwing, nu_redwing, sigma_nearwing, nu_nearwing):

    # Combine the red-wing and near-wing components
    sigma_combined = np.concatenate((sigma_nearwing, sigma_redwing))
    nu_combined    = np.concatenate((nu_nearwing, nu_redwing))
    
    sigma_combined = sigma_combined[np.argsort(nu_combined)]
    nu_combined    = nu_combined[np.argsort(nu_combined)]

    return sigma_combined, nu_combined

def line_profile(nu, nu_0, gamma_L, gamma_G):
    
    # Gandhi et al. (2020b)
    u = (nu - nu_0) / gamma_G
    a = gamma_L / gamma_G

    return Faddeeva(u + 1j*a).real / (gamma_G*np.sqrt(np.pi))

def load_table_nearwing(file):

    with open(file, 'r') as fp:
        
        for i, line in enumerate(fp):
            if i == 9:
                lambda_0, n_ref, vn_norm, pirf_norm = \
                    np.fromstring(line, count=4, sep=' ')
            if i == 11:
                impact_width, impact_shift = \
                    np.fromstring(line, count=2, sep=' ')
                break

    return lambda_0, n_ref, vn_norm, pirf_norm, impact_width, impact_shift

#temperatures = [500, 600, 725, 1000, 1500, 2000, 2500, 3000]
#pressures    = [0.0001, 0.001, 0.01, 0.1, 1., 10., 31.6227766, 100., 316.22776602, 1000.]

#temperatures = [1250]
#pressures    = [0.0001, 0.001, 0.01, 0.1, 1., 3.16227766, 10., 31.6227766, 100., 316.22776602, 1000.]
pressures    = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]

wave_pRT_grid = np.loadtxt('./data/wlen_petitRADTRANS.dat').T

n_max = 1e19
max_Voigt_sep = 250

for P in pressures:
    for T in temperatures:

        n = P / (k_B*T) * 1e6
        print('T={:.0f} K, P={:.6f} bar, n={:.2e} cm^-3'.format(T, P, n))

        # --- D1 line -----------------------
        lambda_0, n_ref, vn_norm, pirf_norm, impact_width, impact_shift = \
            load_table_nearwing('/net/lem/data2/regt/K_I_opacities/tableD1_KHe_4p5s_{:.0f}_1e19.omg'.format(T))
        lambda_0 = 1243.56747315*10

        nu_0 = 1e8/lambda_0
        #n_ref = 1e22
        #n_ref = 1e19

        if n < n_max:
            path = '/net/lem/data2/regt/K_I_opacities/D1/T{:.0f}_P{:.6f}/{}.omg'
            nu_lorentz, sigma_lorentz   = np.loadtxt(path.format(T, P, 'lorentz_out')).T
            nu_nearwing, sigma_nearwing = np.loadtxt(path.format(T, P, 'sigma_out')).T

            nu_lorentz  += nu_0
            nu_nearwing += nu_0

        nu_0 += impact_shift * n/n_ref

        gamma_L = impact_width * n/n_ref
        gamma_G = np.sqrt((2*k_B*T)/(39.0983*amu)) * nu_0/c

        if n < n_max:
            sigma_lorentz = pirf_norm * line_profile(nu_lorentz, nu_0, gamma_L, gamma_G)
            sigma_combined_D1, nu_combined_D1 = nearwing_Lorentz_junction(
                    sigma_lorentz, nu_lorentz, sigma_nearwing, nu_nearwing, nu_0
                    )
        else:
            # Use Lorentzian profile only
            nu_combined_D1 = 1/wave_pRT_grid
            gamma_V = 0.5436*gamma_L + np.sqrt(0.2166*gamma_L**2 + gamma_G**2)

            nu_combined_D1 = nu_combined_D1[
                np.abs(nu_combined_D1 - nu_0) < max_Voigt_sep*gamma_V
                ]
            sigma_combined_D1 = pirf_norm * line_profile(nu_combined_D1, nu_0, gamma_L, gamma_G)

        plt.plot(nu_combined_D1-nu_0, sigma_combined_D1)
        plt.plot(nu_lorentz-nu_0, sigma_lorentz)
        plt.plot(nu_nearwing-nu_0, sigma_nearwing)
        plt.yscale('log')
        plt.savefig(f'opacities_{P}.pdf')
        plt.close()

        sigma_combined_D1 *= 1e-6

        # Interpolate onto pRT wavelength grid
        interp_func_D1 = interp1d(
            1e7/nu_combined_D1, sigma_combined_D1, 
            fill_value=0, bounds_error=False
            )

        sigma_combined_D1D2 = interp_func_D1(wave_pRT_grid*1e7)

        # Add the other lines and save cross-sections
        _, sigma_other_lines = \
            np.loadtxt('/net/lem/data2/regt/K_I_opacities/sigma_{:.0f}.K_{:.6f}bar.dat'.format(T,P)).T
        
        #sigma_tot = sigma_other_lines + sigma_combined_D1D2
        sigma_tot = sigma_other_lines
        np.savetxt(
            '/net/lem/data2/regt/K_I_opacities_tot/sigma_{:.0f}.K_{:.6f}bar.dat'.format(T,P), 
            np.column_stack((wave_pRT_grid, sigma_tot))
            )