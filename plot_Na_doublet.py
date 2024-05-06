import matplotlib.pyplot as plt
import numpy as np

temperatures = np.array([500, 600, 725, 1000, 1500, 2000, 2500, 3000])
pressures    = 10**np.arange(-6,3+1e-6,1)

wave_pRT_grid = np.loadtxt('./data/wlen_petitRADTRANS.dat').T
path_base = '/net/lem/data2/regt/Na_I_opacities_recomputed_pRT_grid/'

wave_mask = (wave_pRT_grid > 0.4e-4) & (wave_pRT_grid < 1.5e-4)

P = 0.1
doublet_i = 'D2'

Z = []
for T in temperatures:
    path_i = path_base+'{}/T{:.0f}_P{:.6f}.dat'.format(doublet_i, T, P)
    opa = np.loadtxt(path_i)
    Z.append(opa[wave_mask])

Z = np.array(Z)
Z_min = Z[Z>0].min()
#Z[Z==0] = Z_min
Z[Z==0] = 1e-250

idx = 5

X = (wave_pRT_grid[wave_mask]*1e7)[None,::idx] * np.ones((len(temperatures),1))
Y = temperatures[:,None] * np.ones((1,Z[:,::idx].shape[1]))

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

from scipy.interpolate import interp1d
interp_Z = interp1d(
    x=np.log10(temperatures), y=np.log10(Z), axis=0, 
    #bounds_error=False, fill_value='extrapolate', kind='slinear'
    bounds_error=False, fill_value=(np.log10(Z)[0], np.log10(Z)[-1]), kind='slinear'#kind='linear'
    )
Z_fine = 10**interp_Z(np.log10(temperatures_fine))
Z_fine[Z_fine<=Z_min] = 0
Z[Z<=Z_min] = 0

X_fine = (wave_pRT_grid[wave_mask]*1e7)[None,::idx] * np.ones((len(temperatures_fine),1))
Y_fine = temperatures_fine[:,None] * np.ones((1,Z_fine[:,::idx].shape[1]))

fig, ax = plt.subplots(figsize=(12,6), ncols=2, sharex=True, sharey=True)
ax[0].pcolormesh(X, Y, np.log10(Z[:,::idx]), rasterized=True, vmin=-30, vmax=-10)
ax[1].pcolormesh(X_fine, Y_fine, np.log10(Z_fine[:,::idx]), rasterized=True, vmin=-30, vmax=-10)
ax[1].set(ylim=(50,3300))
plt.tight_layout()
plt.savefig('./plots/{}_interp.pdf'.format(doublet_i))

Z_fine[Z_fine<=Z_min] = 1e-250
Z[Z<=Z_min] = 1e-250

interp_Z = interp1d(
    x=np.log10(temperatures_fine), y=np.log10(Z_fine), axis=0, bounds_error=False, 
    fill_value=(np.log10(Z_fine)[0], np.log10(Z_fine)[-1]), kind='linear'
)

fig, ax = plt.subplots(figsize=(12,5), ncols=2, gridspec_kw={'width_ratios':[0.7,0.3]}, sharey=True)
for ax_i in ax:
    for i in range(len(temperatures)):
        ax_i.plot(wave_pRT_grid[wave_mask]*1e7, Z[i], color=plt.get_cmap('RdBu_r')(temperatures[i]/3000), zorder=1)

        #'''
        ax_i.plot(
            wave_pRT_grid[wave_mask]*1e7, 10**interp_Z(np.log10(temperatures[i])), 
            color=plt.get_cmap('RdBu_r')(temperatures[i]/3000), ls='--', zorder=2
        )
        #'''

    '''
    for i in range(len(temperatures_fine)):
        ax_i.plot(
            wave_pRT_grid[wave_mask]*1e7, Z_fine[i], 
            color=plt.get_cmap('RdBu_r')(temperatures_fine[i]/3000), 
            ls='--', zorder=2
            )
    '''

ax[0].set(yscale='log')
ax[1].set(yscale='log', xlim=(583,600), ylim=(1e-26,1e-8))
plt.tight_layout()
plt.savefig('./plots/{}_interp_cs.pdf'.format(doublet_i))

temperatures = np.arange(250,4000,250)
temperatures = np.arange(100,1000,50)
fig, ax = plt.subplots(figsize=(12,5), ncols=2, gridspec_kw={'width_ratios':[0.7,0.3]})
for ax_i in ax:
    for i in range(len(temperatures)):
        ax_i.plot(
            wave_pRT_grid[wave_mask]*1e7, 10**interp_Z(np.log10(temperatures[i])), 
            color=plt.get_cmap('RdBu_r')(temperatures[i]/temperatures.max()), ls='-', zorder=2, label=temperatures[i]
        )

ax[0].legend()

ax[0].set(yscale='log', ylim=(1e-26,1e-8))
if doublet_i == 'D1':
    xlim = (589.68,589.85)
elif doublet_i == 'D2':
    xlim = (589.,589.35)
ax[1].set(yscale='log', xlim=xlim, ylim=(1e-20,1e-11))
plt.tight_layout()
plt.savefig('./plots/{}_interp_cs_T_grid.pdf'.format(doublet_i))