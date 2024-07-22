import numpy as np
import matplotlib.pyplot as plt

from scipy.special import wofz as Faddeeva
from scipy.interpolate import interp1d

import glob
import os

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

temperatures      = np.array([500, 600, 725, 1000, 1500, 2000, 2500, 3000])
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
#pressures = 10**np.array([0])

wave_pRT_grid = np.loadtxt('./data/wlen_petitRADTRANS.dat').T

#path_base = '/net/lem/data2/regt/Na_I_opacities_recomputed_pRT_grid/'
path_base = '/net/lem/data2/regt/Na_I_recomputed_3/'
path_table_D1 = path_base + 'TABLES_D1_NaH2_2017/T{:.0f}/table_nearwing_{:.0f}.omg'
path_table_D2 = path_base + 'TABLES_D2_NaH2_2017/tableD2_NaH2_{:.0f}_1e21_FS17.omg'

path_opa = path_base + '{}/T{:.0f}_P{:.6f}/{}.omg'

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
# D1, D2
doublet = {
    'D1': (500, 7.799, True), # Index, log(gamma_N), redwing
    'D2': (499, 7.798, False),
}

for doublet_i, (idx_i, gamma_N_i, redwing_i) in doublet.items():

    for P in pressures:

        sigma_combined_arr = []

        for T in temperatures:

            path_output = path_base + '{}/T{:.0f}_P{:.6f}.dat'.format(doublet_i, T, P)
            try:
                sigma_combined = np.loadtxt(path_output)
                sigma_combined_arr.append(sigma_combined)
                continue

            except FileNotFoundError:
                pass

            # Compute density to determine if Lorentzian should be used
            n = P / (k_B*T) * 1e6
            print('T={:.0f} K, P={:.6f} bar, n={:.2e} cm^-3'.format(T, P, n))

            # Load info from table
            if doublet_i == 'D1':
                path_table_i = path_table_D1.format(T, T)
            elif doublet_i == 'D2':
                path_table_i = path_table_D2.format(T)
            
            lambda_0, n_ref, vn_norm, pirf_norm, impact_width, impact_shift = \
                load_table_nearwing(path_table_i)
            nu_0 = 1e8/lambda_0
            
            # Load pre-computed opacities
            nu_lorentz, sigma_lorentz   = np.loadtxt(path_opa.format(doublet_i, T, P, 'lorentz_out')).T
            nu_nearwing, sigma_nearwing = np.loadtxt(path_opa.format(doublet_i, T, P, 'sigma_out')).T
            
            nu_lorentz  += nu_0
            nu_nearwing += nu_0
            
            if redwing_i:
                nu_redwing, sigma_redwing = np.loadtxt(
                    glob.glob(path_opa.format(doublet_i, T, P, 'redwing*'))[0]
                    ).T
                nu_redwing += nu_0

            # Apply impact shift to line center
            nu_0 += impact_shift * n/n_ref

            # Line-broadening parameters
            gamma_L = impact_width * n/n_ref
            gamma_G = np.sqrt((2*k_B*T)/trans.mass) * nu_0/c

            gamma_N = 10**gamma_N_i / (4*np.pi*c)
            gamma_L += gamma_N

            #print(gamma_G, gamma_L, gamma_L-gamma_N)

            if (n > n_max) or (gamma_N > gamma_L-gamma_N):
                # Use only a Lorentzian component
                nu_combined = 1/wave_pRT_grid

                # Cut the line wing at some separation
                nu_combined = nu_combined[
                    np.abs(nu_combined - nu_0) < max_Voigt_sep
                    ]
                sigma_combined = pirf_norm * line_profile(nu_combined, nu_0, gamma_L, gamma_G)
            
            else:
                # Compute the shifted Lorentzian
                sigma_lorentz = pirf_norm * line_profile(nu_lorentz, nu_0, gamma_L, gamma_G)
                
                # Combine the Lorentz profile and nearwing-component
                sigma_combined, nu_combined = nearwing_Lorentz_junction(
                        sigma_lorentz, nu_lorentz, sigma_nearwing, nu_nearwing, nu_0
                        )
                
                if redwing_i:
                    # Combine the nearwing/Lorentzian with the redwing
                    sigma_redwing *= n/n_ref
                    sigma_combined, nu_combined = redwing_junction(
                        sigma_redwing, nu_redwing, sigma_combined, nu_combined
                        )

            # Normalize the line profile, so that integral equals 1
            sigma_combined  = sigma_combined[np.argsort(nu_combined)]
            nu_combined     = nu_combined[np.argsort(nu_combined)]
            sigma_combined /= np.trapz(x=nu_combined, y=sigma_combined)

            # Scale with the integrated line intensity
            trans(P=P, T=T, states=states)
            sigma_combined *= trans.S[idx_i]

            wave_minmax = (1e7/nu_combined.max(), 1e7/nu_combined.min())

            # Interpolate onto the pRT wavelength grid
            interp_func = interp1d(
                1e7/nu_combined, sigma_combined, fill_value=0, bounds_error=False
            )
            sigma_combined = interp_func(wave_pRT_grid*1e7)

            # Save the opacity
            np.savetxt(path_output, sigma_combined)
            sigma_combined_arr.append(sigma_combined)

        # Perform the interpolation to the finer temperature grid

        # Use only the relevant wavelengths
        sigma_combined_arr = np.array(sigma_combined_arr)
        wave_mask          = (sigma_combined_arr > 0).any(axis=0)
        sigma_combined_arr = sigma_combined_arr[:,wave_mask]

        # Set to non-zero values
        sigma_min = sigma_combined_arr[sigma_combined_arr>0].min()
        #sigma_combined_arr[sigma_combined_arr==0] = 1e-250
        sigma_combined_arr[sigma_combined_arr==0] = sigma_min
        #sigma_combined_arr[sigma_combined_arr==0] = sigma_combined_arr[sigma_combined_arr!=0][-2]

        print('Interpolating {} for pressure {:.6f} bar'.format(doublet_i, P))

        interp_func = interp1d(
            x=np.log10(temperatures), y=np.log10(sigma_combined_arr), axis=0, bounds_error=False, 
            # Avoid extrapolation and use the outer values
            fill_value=(np.log10(sigma_combined_arr[0]), np.log10(sigma_combined_arr[-1])), 
            kind='slinear' # Linear spine interpolation
        )
        # Interpolate onto finer T-grid
        sigma_combined_arr_fine = 10**interp_func(np.log10(temperatures_fine))
        sigma_combined_arr_fine[sigma_combined_arr_fine <= sigma_min*1.1] = 0

        fig, ax = plt.subplots(figsize=(12,6))
        for i, T_i in enumerate(temperatures):
            ax.plot(
                wave_pRT_grid[wave_mask]*1e7, sigma_combined_arr[i], label=f'Allard T={T_i}', 
                color=plt.get_cmap('RdBu_r')(T_i/3000)
                )
        for i, T_i in enumerate(temperatures_fine):
            ax.plot(
                wave_pRT_grid[wave_mask]*1e7, sigma_combined_arr_fine[i], label='pRT T={:.0f}'.format(T_i), 
                color=plt.get_cmap('RdBu_r')(T_i/3000), ls='--'
                )
        ax.legend()
        ax.set(ylim=(1e-26,1e-8), yscale='log')
        plt.tight_layout()
        plt.savefig('./plots/Na_doublet_3/{}_P{:.1e}_interp.pdf'.format(doublet_i,P))
        plt.close()
        #import sys; sys.exit()

        # Save the opacities for each wavelength
        for i, T_i in enumerate(temperatures_fine):

            path_output = path_base + \
                'opacities_doublet/{}_T{:.0f}_P{:.6f}.dat'
            
            if doublet_i == 'D1':
                sigma_i = np.zeros_like(wave_pRT_grid)
                path_output = path_output.format(doublet_i, T_i, P)
            elif doublet_i == 'D2':
                # Load the other line and add
                sigma_i = np.loadtxt(path_output.format('D1', T_i, P))
                path_output = path_output.format('D1D2', T_i, P)                

            sigma_i[wave_mask] += sigma_combined_arr_fine[i]
            np.savetxt(path_output, sigma_i)
            del sigma_i

            if doublet_i == 'D2':
                # Delete the D1 file
                os.remove(path_output.replace('D1D2', 'D1'))

        del sigma_combined_arr_fine
