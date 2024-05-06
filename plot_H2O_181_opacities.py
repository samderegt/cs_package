import matplotlib.pyplot as plt
import numpy as np

from petitRADTRANS import Radtrans

atm1 = Radtrans(
    #line_species=['H2O_main_iso'], mode='lbl', lbl_opacity_sampling=1, 
    line_species=['H2O_181'], mode='lbl', lbl_opacity_sampling=1, 
    wlen_bords_micron=[0.3,2.5]
    )

atm2 = Radtrans(
    #line_species=['H2O_pokazatel_main_iso'], mode='lbl', lbl_opacity_sampling=1, 
    line_species=['H2O_181_HotWat78'], mode='lbl', lbl_opacity_sampling=1, 
    wlen_bords_micron=[0.3,2.5]
    )

temperatures = np.arange(500,3000+1e-6,500)
pressures    = [1e-6, 1e-4, 1e-2, 1, 100]
#pressures = [10,100]

wave_range = [(300,2500), (1138,1138.9), (1950,2450), (2406,2411)]

for P in pressures:
    fig, ax = plt.subplots(figsize=(12,11), nrows=4)
    
    for i, T in enumerate(temperatures):

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
            
            label = 'HITRAN: T={:.0f}K'.format(T)
            ax_j.plot(
                wave_micron[mask]*1e3, opa1[mask] * (20*1.66054e-24), 
                c=plt.get_cmap('coolwarm')((i)/len(temperatures)), ls='--', 
                label=label, zorder=1
                )
            ax_j.plot(
                wave_micron[mask]*1e3, opa2[mask] * (20*1.66054e-24), 
                c=plt.get_cmap('coolwarm')((i)/len(temperatures)), zorder=0,
                )
            ax_j.set(yscale='log', xlim=wave_range[j])

    ax[1].legend(ncols=2)
    ax[0].set(ylim=(1e-35,1e-14))
    ax[1].set(ylim=(1e-35,1e-14))
    ax[2].set(ylim=(1e-35,1e-14))
    ax[3].set(ylim=(1e-35,1e-14))

    ax[0].set(title='P={:.0e}bar'.format(P))
    ax[-1].set(
        xlabel='Wavelength (nm)', 
        ylabel=r'Cross-section (cm$^2$)'
        )
    
    plt.tight_layout()
    plt.savefig('./plots/H2O_181_opacities_tot_P{:.6f}.pdf'.format(P))
    #plt.show()
    plt.close()


for T in temperatures:
    fig, ax = plt.subplots(figsize=(12,11), nrows=4)
    
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
            
            label = 'HITRAN: P={:.0e}bar'.format(P)
            ax_j.plot(
                wave_micron[mask]*1e3, opa1[mask] * (20*1.66054e-24), 
                c=plt.get_cmap('viridis_r')((i)/len(pressures)), ls='--', 
                label=label, zorder=1
                )
            ax_j.plot(
                wave_micron[mask]*1e3, opa2[mask] * (20*1.66054e-24), 
                c=plt.get_cmap('viridis_r')((i)/len(pressures)), zorder=0,
                )
            ax_j.set(yscale='log', xlim=wave_range[j])

    ax[1].legend(ncols=2)
    ax[0].set(ylim=(1e-35,1e-14))
    ax[1].set(ylim=(1e-35,1e-14))
    ax[2].set(ylim=(1e-35,1e-14))
    ax[3].set(ylim=(1e-35,1e-14))

    ax[0].set(title='T={:.0e}K'.format(T))
    ax[-1].set(
        xlabel='Wavelength (nm)', 
        ylabel=r'Cross-section (cm$^2$)'
        )
    
    plt.tight_layout()
    plt.savefig('./plots/H2O_181_opacities_tot_T{:.0f}.pdf'.format(T))
    #plt.show()
    plt.close()