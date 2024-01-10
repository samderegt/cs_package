import numpy as np
import matplotlib.pyplot as plt

from petitRADTRANS import Radtrans

amu = 1.66054e-24
'''
atm = Radtrans(
    line_species=['Fe'], 
    mode='lbl', lbl_opacity_sampling=1, 
    wlen_bords_micron=[0.3,1.3]
    )

wave_micron, pRT_opa = atm.plot_opas(
    ['Fe'], temperature=4000, pressure_bar=1, return_opacities=True
    )['Fe']
pRT_opa *= 55.845*amu
'''

wave  = np.loadtxt('./data/wave.dat')
sigma = np.loadtxt('./data/opacities.dat')

plt.figure(figsize=(11,5))
plt.plot(wave, sigma, lw=1, label='computed')
#plt.plot(1e3*wave_micron, pRT_opa, lw=1, label='pRT')
plt.legend()
plt.yscale('log')
plt.show()