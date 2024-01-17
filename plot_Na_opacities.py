import matplotlib.pyplot as plt
import numpy as np

from petitRADTRANS import Radtrans

atm = Radtrans(
    line_species=['Na_allard'], 
    mode='lbl', lbl_opacity_sampling=1, 
    wlen_bords_micron=[0.35,1.2]
    )

atm2 = Radtrans(
    line_species=['Na_recomputed'], 
    mode='lbl', lbl_opacity_sampling=1, 
    wlen_bords_micron=[0.35,1.2]
    )

#temperatures = [500, 600, 725, 1000, 1500, 2000, 2500, 3000]
temperatures = [2300]
#pressures    = [0.0001, 0.001, 0.01, 0.1, 1., 10., 31.6227766, 100., 316.22776602, 1000.]
pressures    = [0.0001, 0.01, 1., 100.]

wave_range = [(360,1200), (587,592.5), (568.3,569.05)]

for T in temperatures:

    fig, ax = plt.subplots(figsize=(12,9), nrows=3)
    for i, P in enumerate(pressures):

        n = P / (1.3807e-16*T*1e-6)
        print('{:.2e}'.format(n))
        if n > 1e21:
            continue

        #wave_cm, sigma = np.loadtxt('/net/lem/data2/regt/Na_I_opacities_tot/sigma_{:.0f}.K_{:.6f}bar.dat'.format(T, P)).T
        #mask = (wave_cm*1e4 > 0.3) & (wave_cm*1e4 < 28)
        #wave_cm = wave_cm[mask]
        #sigma   = sigma[mask]
        #del mask

        wave_micron, opa = atm.plot_opas(
            ['Na_allard'], temperature=T, pressure_bar=P, return_opacities=True
            )['Na_allard']
        
        wave_micron, opa2 = atm2.plot_opas(
            ['Na_recomputed'], temperature=T, pressure_bar=P, return_opacities=True
            )['Na_recomputed']
        
        for j, ax_j in enumerate(ax):
            #mask = (wave_cm*1e7 > wave_range[j][0]-1) & (wave_cm*1e7 < wave_range[j][1]+1)
            #ax_j.plot(
            #    wave_cm[mask]*1e7, sigma[mask], 
            #    c=plt.get_cmap('viridis_r')((i)/len(pressures)), 
            #    label='P={:.6f} bar'.format(P)
            #    )
            
            ax_j.plot(
                wave_micron*1e3, opa * (23*1.66054e-24), 
                c=plt.get_cmap('cividis_r')((i)/len(pressures)), ls='--', 
                label='pRT: P={:.6f} bar'.format(P)
                )

            ax_j.plot(
                wave_micron*1e3, opa2 * (23*1.66054e-24), 
                c=plt.get_cmap('viridis_r')((i)/len(pressures)), 
                label='new: P={:.6f} bar'.format(P)
                )
            
    ax[0].set(yscale='log', xlim=wave_range[0], ylim=(1e-27,1e-10), title='T={:.0f} K'.format(T))
    ax[0].legend(loc='upper right', ncols=2)

    ax[1].set(yscale='log', xlim=wave_range[1], ylim=(1e-23,1e-10))
    ax[2].set(
        yscale='log', xlim=wave_range[2], ylim=(1e-23,1e-15), 
        xlabel='Wavelength (nm)', ylabel=r'Cross-section (cm$^2$)'
        )
    
    plt.tight_layout()
    plt.savefig('./Na_opacities_tot.pdf')
    #plt.show()
    plt.close()