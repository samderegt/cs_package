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

temperatures = [500, 600, 725, 1000, 1500, 2000, 2500, 3000]
pressures    = 10**np.arange(-6,3+1e-6,1)

wave_pRT_grid = np.loadtxt('./data/wlen_petitRADTRANS.dat').T

n_max = 1e21
max_Voigt_sep = 4500

from opacities import Transitions, States
states = States('./data/Na_I_states.txt')
trans = Transitions(
    #'./data/Na_I_transitions.txt', 
    './data/Na_I_transitions_ck.txt', 
    mass=22.989769*amu, E_ion=41449.451, 
    is_alkali=True, only_valid=True
    )

for P in pressures:
    for T in temperatures:

        n = P / (k_B*T) * 1e6
        print('T={:.0f} K, P={:.6f} bar, n={:.2e} cm^-3'.format(T, P, n))

        # --- D1 resonance line -----------------------
        lambda_0, n_ref, vn_norm, pirf_norm, impact_width, impact_shift = \
            load_table_nearwing(f'/net/lem/data2/regt/Na_I_opacities_recomputed/TABLES_D1_NaH2_2017/T{T}/table_nearwing_{T}.omg')

        path = '/net/lem/data2/regt/Na_I_opacities_recomputed/D1/T{:.0f}_P{:.6f}/{}.omg'
        nu_lorentz, sigma_lorentz   = np.loadtxt(path.format(T, P, 'lorentz_out')).T
        nu_nearwing, sigma_nearwing = np.loadtxt(path.format(T, P, 'sigma_out')).T
        nu_redwing, sigma_redwing   = np.loadtxt(glob.glob(path.format(T, P, 'redwing*'))[0]).T

        nu_0 = 1e8/lambda_0
        nu_lorentz  += nu_0
        nu_nearwing += nu_0
        nu_redwing  += nu_0

        nu_0 += impact_shift * n/n_ref

        gamma_L = impact_width * n/n_ref
        gamma_G = np.sqrt((2*k_B*T)/(22.989769*amu)) * nu_0/c

        gamma_N = 10**(7.799) / (4*np.pi*c)
        gamma_L += gamma_N

        print(gamma_G, gamma_L, gamma_L-gamma_N)

        sigma_lorentz = pirf_norm * line_profile(nu_lorentz, nu_0, gamma_L, gamma_G)
        sigma_combined, nu_combined = nearwing_Lorentz_junction(
                sigma_lorentz, nu_lorentz, sigma_nearwing, nu_nearwing, nu_0
                )

        sigma_redwing *= n/n_ref
        sigma_combined_D1, nu_combined_D1 = redwing_junction(sigma_redwing, nu_redwing, sigma_combined, nu_combined)
        if (n > n_max) or (gamma_N > gamma_L-gamma_N):
            # Use Lorentzian profile only
            nu_combined_D1 = 1/wave_pRT_grid
            gamma_V = 0.5436*gamma_L + np.sqrt(0.2166*gamma_L**2 + gamma_G**2)

            nu_combined_D1 = nu_combined_D1[
                #np.abs(nu_combined_D1 - nu_0) < max_Voigt_sep*gamma_V
                np.abs(nu_combined_D1 - nu_0) < max_Voigt_sep
                ]
            sigma_combined_D1 = pirf_norm * line_profile(nu_combined_D1, nu_0, gamma_L, gamma_G)

        # Normalize the line profile, so that integral equals 1
        sigma_combined_D1 = sigma_combined_D1[np.argsort(nu_combined_D1)]
        nu_combined_D1    = nu_combined_D1[np.argsort(nu_combined_D1)]
        sigma_combined_D1 /= np.trapz(x=nu_combined_D1, y=sigma_combined_D1)
        
        # Scale with the integrated line intensity
        trans(P=P, T=T, states=states)
        #sigma_combined_D1 *= trans.S[471]
        sigma_combined_D1 *= trans.S[500]

        # --- D2 resonance line -----------------------
        lambda_0, n_ref, vn_norm, pirf_norm, impact_width, impact_shift = \
            load_table_nearwing(f'/net/lem/data2/regt/Na_I_opacities_recomputed/TABLES_D2_NaH2_2017/tableD2_NaH2_{T}_1e21_FS17.omg')

        path = '/net/lem/data2/regt/Na_I_opacities_recomputed/D2/T{:.0f}_P{:.6f}/{}.omg'
        nu_lorentz, sigma_lorentz   = np.loadtxt(path.format(T, P, 'lorentz_out')).T
        nu_nearwing, sigma_nearwing = np.loadtxt(path.format(T, P, 'sigma_out')).T

        nu_0 = 1e8/lambda_0
        nu_lorentz  += nu_0
        nu_nearwing += nu_0

        nu_0 += impact_shift * n/n_ref

        gamma_L = impact_width * n/n_ref
        gamma_G = np.sqrt((2*k_B*T)/(22.989769*amu)) * nu_0/c

        gamma_N = 10**(7.798) / (4*np.pi*c)
        gamma_L += gamma_N

        print(gamma_G, gamma_L, gamma_L-gamma_N)

        sigma_lorentz = pirf_norm * line_profile(nu_lorentz, nu_0, gamma_L, gamma_G)
        sigma_combined_D2, nu_combined_D2 = nearwing_Lorentz_junction(
                sigma_lorentz, nu_lorentz, sigma_nearwing, nu_nearwing, nu_0
                )
        if (n > n_max) or (gamma_N > gamma_L-gamma_N):
            # Use Lorentzian profile only
            nu_combined_D2 = 1/wave_pRT_grid
            gamma_V = 0.5436*gamma_L + np.sqrt(0.2166*gamma_L**2 + gamma_G**2)

            nu_combined_D2 = nu_combined_D2[
                #np.abs(nu_combined_D2 - nu_0) < max_Voigt_sep*gamma_V
                np.abs(nu_combined_D2 - nu_0) < max_Voigt_sep
                ]
            sigma_combined_D2 = pirf_norm * line_profile(nu_combined_D2, nu_0, gamma_L, gamma_G)

        # Normalize the line profile, so that integral equals 1
        sigma_combined_D2 = sigma_combined_D2[np.argsort(nu_combined_D2)]
        nu_combined_D2    = nu_combined_D2[np.argsort(nu_combined_D2)]
        sigma_combined_D2 /= np.trapz(x=nu_combined_D2, y=sigma_combined_D2)
        
        # Scale with the integrated line intensity
        trans(P=P, T=T, states=states)
        #sigma_combined_D2 *= trans.S[470]
        sigma_combined_D2 *= trans.S[499]

        # -----------------------------------------
        # Interpolate onto pRT wavelength grid
        interp_func_D1 = interp1d(
            1e7/nu_combined_D1, sigma_combined_D1, 
            fill_value=0, bounds_error=False
            )
        interp_func_D2 = interp1d(
            1e7/nu_combined_D2, sigma_combined_D2, 
            fill_value=0, bounds_error=False
            )

        sigma_combined_D1D2 = interp_func_D1(wave_pRT_grid*1e7) + \
            interp_func_D2(wave_pRT_grid*1e7)

        # Add the other lines and save cross-sections
        _, sigma_other_lines = \
            np.loadtxt('/net/lem/data2/regt/Na_I_opacities_recomputed/opacities_wo_doublet/sigma_{:.0f}.K_{:.6f}bar.dat'.format(T,P)).T
        
        # Save wavelength and cross-section
        sigma_tot = sigma_other_lines + sigma_combined_D1D2
        np.savetxt(
            '/net/lem/data2/regt/Na_I_opacities_recomputed/sigma_{:.0f}.K_{:.6f}bar.dat'.format(T,P), 
            np.column_stack((wave_pRT_grid, sigma_tot))
            )