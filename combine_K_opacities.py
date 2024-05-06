import numpy as np
import matplotlib.pyplot as plt

from scipy.special import wofz as Faddeeva
from scipy.interpolate import interp1d

e      = 4.8032e-10    # Electron charge, cm^3/2 g^1/2 s^-1
m_e    = 9.1094e-28    # Electron mass, g
c2     = 1.4387769     # cm K
k_B    = 1.3807e-16    # cm^2 g s^-2 K^-1
c      = 2.99792458e10 # cm s^-1
N_A    = 6.02214076e23 # mol^-1
mass_H = 1.6735575e-24 # g
amu    = 1.66054e-24   # g

def nearwing_Lorentz_junction(
        sigma_lorentz, nu_lorentz, sigma_nearwing, nu_nearwing, nu_0, max_nu_offset=60
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

#temperatures = [1250]
#temperatures = [500,725,1000,1250,1500,2000,2500,3000]
#temperatures = [500,725,1000,1500,2000,2500,3000]
temperatures = np.array([
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
#pressures = 10**np.array([-6,-4,-2,0,1,2],dtype=float)

wave_pRT_grid = np.loadtxt('./data/wlen_petitRADTRANS.dat').T

# Load Nicole's table
#all_T, all_w, all_d = np.loadtxt(
#    './data/param_D1_KHe_4p5s.dat', skiprows=2, usecols=(0,1,2), 
#    ).T

n_max = 1e19
#max_Voigt_sep = 250
max_Voigt_sep = 4500

from opacities import Transitions, States, BroadeningPowerLaw
broad = BroadeningPowerLaw(
    A_w=[0.121448], b_w=[0.531718], A_d=[0.000462539], b_d=[1.07284]
    )
states = States('./data/K_I_states.txt')
trans = Transitions(
    './data/K_I_transitions.txt', mass=39.0983*amu, E_ion=35009.8140, 
    is_alkali=True, only_valid=True
    )

for T in temperatures:
    
    plt.figure(figsize=(7,4))

    for i, P in enumerate(pressures):

        n = P / (k_B*T) * 1e6
        print('T={:.0f} K, P={:.6f} bar, n={:.2e} cm^-3'.format(T, P, n))

        # --- D1 line -----------------------
        lambda_0, n_ref, vn_norm, pirf_norm, impact_width, impact_shift = \
            load_table_nearwing('/net/lem/data2/regt/K_I_opacities_new/D1/tableD1_KHe_4p5s_1250_1e19.omg')

        #n_ref = 1e20
        gamma_L, impact_shift = broad(P, T)

        nu_0 = 1e8/lambda_0
        nu_0 += impact_shift

        gamma_G = np.sqrt((2*k_B*T)/(39.0983*amu)) * nu_0/c
        gamma_N = 10**(7.790) / (4*np.pi*c)
        gamma_L += gamma_N

        if n < n_max:
            path = '/net/lem/data2/regt/K_I_opacities_new/D1/T1250_P{:.6f}/{}.omg'
            nu_lorentz, sigma_lorentz   = np.loadtxt(path.format(P, 'lorentz_out')).T
            nu_nearwing, sigma_nearwing = np.loadtxt(path.format(P, 'sigma_out')).T

            #nu_lorentz  += nu_0
            nu_lorentz += 1e8/lambda_0
            #nu_nearwing += nu_0
            nu_nearwing += 1e8/lambda_0

            if gamma_N > gamma_L-gamma_N:
                # Use Lorentzian profile only
                nu_lorentz = 1/wave_pRT_grid
                nu_lorentz = nu_lorentz[np.abs(nu_lorentz - nu_0) < max_Voigt_sep]

            sigma_lorentz = pirf_norm * line_profile(nu_lorentz, nu_0, gamma_L, gamma_G)
            if gamma_N > gamma_L-gamma_N:
                sigma_combined_D1 = sigma_lorentz
                nu_combined_D1    = nu_lorentz
            else:
                sigma_combined_D1, nu_combined_D1 = nearwing_Lorentz_junction(
                        sigma_lorentz, nu_lorentz, sigma_nearwing, nu_nearwing, nu_0
                        )
            
        else:
            # Use Lorentzian profile only
            nu_combined_D1 = 1/wave_pRT_grid
            gamma_V = 0.5436*gamma_L + np.sqrt(0.2166*gamma_L**2 + gamma_G**2)

            nu_combined_D1 = nu_combined_D1[
                #np.abs(nu_combined_D1 - nu_0) < max_Voigt_sep*gamma_V
                np.abs(nu_combined_D1 - nu_0) < max_Voigt_sep
                ]
            sigma_combined_D1 = pirf_norm * line_profile(nu_combined_D1, nu_0, gamma_L, gamma_G)

        sigma_combined_D1 = sigma_combined_D1[np.argsort(nu_combined_D1)]
        nu_combined_D1    = nu_combined_D1[np.argsort(nu_combined_D1)]

        # Normalize the line profile, so that integral equals 1
        sigma_combined_D1 /= np.trapz(x=nu_combined_D1, y=sigma_combined_D1)
        
        # Scale with the integrated line intensity
        trans(P=P, T=T, states=states)
        sigma_combined_D1 *= trans.S[1096]

        l = plt.plot(
            1e7/nu_combined_D1, sigma_combined_D1, 
            c=plt.get_cmap('viridis_r')(i/len(pressures)), alpha=0.2, 
            )
        
        # Interpolate onto pRT wavelength grid
        interp_func_D1 = interp1d(
            1e7/nu_combined_D1, sigma_combined_D1, 
            fill_value=0, bounds_error=False
            )

        sigma_combined_D1D2 = interp_func_D1(wave_pRT_grid*1e7)

        # Add the other lines and save cross-sections
        _, sigma_other_lines = \
            np.loadtxt('/net/lem/data2/regt/K_I_opacities_new/opacities_wo_D1/sigma_{:.0f}.K_{:.6f}bar.dat'.format(T,P)).T
        
        sigma_tot = sigma_other_lines + sigma_combined_D1D2

        mask = (wave_pRT_grid*1e7 > 1200) & (wave_pRT_grid*1e7 < 1300)
        plt.plot(wave_pRT_grid[mask]*1e7, sigma_tot[mask], c=l[0].get_color())
        plt.plot(wave_pRT_grid[mask]*1e7, sigma_other_lines[mask], c=l[0].get_color(), alpha=0.2)
    
        np.savetxt(
            '/net/lem/data2/regt/K_I_opacities_new/sigma_{:.0f}.K_{:.6f}bar.dat'.format(T,P), 
            np.column_stack((wave_pRT_grid, sigma_tot))
            )
    
    plt.axvline(1e7/8041.38112, linewidth=1, color='k', linestyle='--')
    plt.axvline(1e7/7983.67489, linewidth=1, color='k', linestyle='--')
    plt.yscale('log'); plt.ylim(1e-28,1e-14); plt.xlim(1200,1300)
    plt.savefig('./plots/opacities_T{:.0f}.pdf'.format(T))

    plt.yscale('log'); plt.ylim(1e-28,1e-14); plt.xlim(1240,1256)
    plt.savefig('./plots/opacities_T{:.0f}_zoom_in.pdf'.format(T))
    plt.close()