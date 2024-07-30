import matplotlib.pyplot as plt
import numpy as np

from petitRADTRANS import Radtrans

#prefix = 'H2O_181_'
#line_species = ['H2O_181', 'H2O_181_HotWat78']
#mass = 20

prefix = 'Fe'
line_species = ['Fe', 'Fe_Sam']
mass = 55.845

#prefix = 'H2O'
#line_species = ['H2O_pokazatel_main_iso', 'H2O_pokazatel_main_iso_Sam']
#mass = 18

#prefix = 'VO'
#line_species = ['VO_ExoMol_McKemmish', 'VO_HyVO_main_iso']
#mass = 51+16

#prefix = 'Na_recomputed_3'
#line_species = ['Na_allard', 'Na_allard_recomputed']
#mass = 23

#prefix = 'CO'
#line_species = ['CO_high', 'CO_high_Sam']; mass = 12+16
#line_species = ['CO_36', 'CO_36_high']; mass = 13+16
#line_species = ['CO_36_high', 'CO_36_high_Sam']; mass = 13+16
#temperatures = np.arange(250,3000+1e-6,250)
#temperatures = np.arange(1250,1500+1e-6,250)
#temperatures = np.arange(250,3000+1e-6,750)
#temperatures = [1700]
temperatures = np.arange(1500,3500,500)
#temperatures = [300,400,500,600,700,800,900,1000,1200,1400]

#temperatures = [300,400,500,600,700]
#temperatures = [600,700,800,900,1000,1200,1400]
#temperatures = [1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000]
pressures    = [0.1]
#wave_range   = [(2320,2370), (2330,2340), (2320,2327), (2334,2337)]
#wave_range   = [(2342,2500), (2400,2405), (2405,2410), (2410,2415), ]
#wave_range   = [(2490,2495), (2495,2500), (2485,2490), (2470,2475)]

#wave_min, wave_max = 2280, 2500
wave_min, wave_max = 1100, 1200
d_wave = 2.5
wave_range = [
    (x,x+d_wave) for x in np.arange(wave_min,wave_max,d_wave)
]
#wave_range   = [(2265,2273)]*4
#wave_range   = [(2230,2240)]*4

### !!!!!
# (2470,2475) # 13CO
### !!!!!

'''
#temperatures = np.arange(500,3000+1e-6,500)
temperatures = [1500, 2000, 2500, 3000]
pressures    = [1]

#wave_range = [(300,2500), (1138,1138.9), (1950,2450), (2406,2411)]
#ylim = (1e-35,1e-14)
wave_range = [(500,1500), (1300,1308), (1317,1325), (608.3,610)]
'''
ylim = None#(1e-23,1e-18)
#ylim = (1e-24,1e-20)
#ylim = (1e-36,1e-10)
lbl_opacity_sampling = 1

atm1 = Radtrans(
    line_species=[line_species[0]], mode='lbl', 
    lbl_opacity_sampling=lbl_opacity_sampling, 
    wlen_bords_micron=[wave_min*1e-3,wave_max*1e-3]
    )

atm2 = Radtrans(
    line_species=[line_species[1]], mode='lbl', 
    lbl_opacity_sampling=lbl_opacity_sampling, 
    wlen_bords_micron=[wave_min*1e-3,wave_max*1e-3]
    )

import os
if not os.path.exists(f'./plots/{prefix}'):
    os.makedirs(f'./plots/{prefix}')

for P in pressures:
    fig, ax = plt.subplots(figsize=(12,11*len(wave_range)/4), nrows=len(wave_range))
    
    for i, T in enumerate(temperatures):

        n = P / (1.3807e-16*T*1e-6)
        #print('{:.2e}'.format(n))

        wave_micron, opa1 = atm1.plot_opas(
            atm1.line_species, temperature=T, pressure_bar=P, return_opacities=True
            )[atm1.line_species[0]]
        wave_micron, opa2 = atm2.plot_opas(
            atm2.line_species, temperature=T, pressure_bar=P, return_opacities=True
            )[atm2.line_species[0]]
        #opa1 *= np.nan
        
        for j, ax_j in enumerate(ax):

            mask = (wave_micron*1e3 > wave_range[j][0]-1) & \
                (wave_micron*1e3 < wave_range[j][1]+1)
            
            label = '{} | T={:.0f}K'.format(line_species[0], T)
            ax_j.plot(
                wave_micron[mask]*1e3, opa1[mask] * (mass*1.66054e-24), 
                c=plt.get_cmap('coolwarm')(i/(len(temperatures)-1)), ls='--', 
                label=label, zorder=0, #alpha=0.5
                )
            ax_j.plot(
                wave_micron[mask]*1e3, opa2[mask] * (mass*1.66054e-24), 
                c=plt.get_cmap('coolwarm')(i/(len(temperatures)-1)), zorder=1,
                )
            ax_j.set(yscale='log', xlim=wave_range[j])

    ax[1].legend(ncols=2)
    for ax_i in ax:
        ax_i.set(ylim=ylim)

    ax[0].set(title='P={:.0e}bar'.format(P))
    ax[-1].set(
        xlabel='Wavelength (nm)', 
        ylabel=r'Cross-section (cm$^2$)'
        )
    
    plt.tight_layout()
    plt.savefig('./plots/{}/opacities_tot_P{:.6f}.pdf'.format(prefix, P))
    #plt.show()
    plt.close()

import sys; sys.exit()

for T in temperatures:
    fig, ax = plt.subplots(figsize=(12,11*len(wave_range)/4), nrows=len(wave_range))
    
    for i, P in enumerate(pressures):

        n = P / (1.3807e-16*T*1e-6)
        #print('{:.2e}'.format(n))

        wave_micron, opa1 = atm1.plot_opas(
            atm1.line_species, temperature=T, pressure_bar=P, return_opacities=True
            )[atm1.line_species[0]]
        wave_micron, opa2 = atm2.plot_opas(
            atm2.line_species, temperature=T, pressure_bar=P, return_opacities=True
            )[atm2.line_species[0]]
        
        for j, ax_j in enumerate(ax):

            mask = (wave_micron*1e3 > wave_range[j][0]-1) & \
                (wave_micron*1e3 < wave_range[j][1]+1)
            
            label = '{} | P={:.0e}bar'.format(line_species[0], P)
            ax_j.plot(
                wave_micron[mask]*1e3, opa1[mask] * (mass*1.66054e-24), 
                c=plt.get_cmap('viridis_r')((i)/len(pressures)), ls='--', 
                label=label, zorder=0, alpha=0.5
                )
            ax_j.plot(
                wave_micron[mask]*1e3, opa2[mask] * (mass*1.66054e-24), 
                c=plt.get_cmap('viridis_r')((i)/len(pressures)), zorder=1,
                )
            ax_j.set(yscale='log', xlim=wave_range[j])

    ax[1].legend(ncols=2)
    for ax_i in ax:
        ax_i.set(ylim=ylim)

    ax[0].set(title='T={:.0e}K'.format(T))
    ax[-1].set(
        xlabel='Wavelength (nm)', 
        ylabel=r'Cross-section (cm$^2$)'
        )
    
    plt.tight_layout()
    plt.savefig('./plots/{}/opacities_tot_T{:.0f}.pdf'.format(prefix, T))
    #plt.show()
    plt.close()