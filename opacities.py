import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
#os.environ['OMP_NUM_THREADS'] = '1'

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from scipy.special import wofz as Faddeeva

e      = 4.8032e-10    # Electron charge, cm^3/2 g^1/2 s^-1
m_e    = 9.1094e-28    # Electron mass, g
c2     = 1.4387769     # cm K
k_B    = 1.3807e-16    # cm^2 g s^-2 K^-1
c      = 2.99792458e10 # cm s^-1
N_A    = 6.02214076e23 # mol^-1
mass_H = 1.6735575e-24 # g
amu    = 1.66054e-24   # g

class States:
    def __init__(self, NIST_tsv_file):

        self.load_file(NIST_tsv_file)

    def load_file(self, file):

        d = np.loadtxt(file, dtype=str, skiprows=1, usecols=(0,2))

        g, E = [], []
        for g_i, E_i in d:

            if g_i == '""':
                continue
            g.append(float(g_i))
            E.append(float(E_i.replace('"', '')))
        
        self.g = np.array(g)
        self.E = np.array(E)

    def partition_function(self, T):

        Q = np.sum(self.g * np.exp(-c2*self.E/T))
        return Q
    
class Transitions:
    def __init__(
            self, VALD_short_format_file, 
            mass, E_ion, Z=0, 
            is_alkali=False, only_trans_with_VALD_damping=True
            ):
        
        self.is_alkali  = is_alkali
        self.only_trans_with_VALD_damping = only_trans_with_VALD_damping
        self.load_file(VALD_short_format_file)

        # Mass of atom [g]
        self.mass = mass
        # Ionization energy [cm^-1]
        self.E_ion = E_ion
        self.Z = Z # Electron charge

        # Broadening parameters
        self.gamma_G   = np.zeros_like(self.nu_0)
        self.gamma_L   = np.zeros_like(self.nu_0)
        self.gamma_vdW = np.zeros_like(self.nu_0)

    def load_file(self, file):

        self.d = np.genfromtxt(file, delimiter=',', dtype=float, 
            skip_header=2, usecols=(1,2,3,4,5,6), 
            invalid_raise=False
            )
        
        self.nu_0   = self.d[:,0]
        self.E_low  = self.d[:,1]
        self.E_high = self.E_low + self.nu_0
        
        # Oscillator strength
        self.log_gf = self.d[:,2]
        self.gf     = 10**self.log_gf

        # log10 Radiative, Stark and vdW-damping constants
        # [s^-1], [s^-1 cm^3], [s^-1 cm^3]
        self.damping = self.d[:,3:]

        # Make a copy of the initial wavenumbers
        self.nu_0_init = np.copy(self.nu_0)

        # Check that all transitions have a vdW- and 
        # natural broadening coefficient
        mask_valid = (self.damping[:,0]!=0) & (self.damping[:,2]!=0)
        
        if self.is_alkali:
            # Broadening coefficients are derived for alkalis
            return
        
        if self.only_trans_with_VALD_damping:
            # Remove the invalid transitions
            self.nu_0   = self.nu_0[mask_valid]
            self.E_low  = self.E_low[mask_valid]
            self.E_high = self.E_high[mask_valid]

            self.log_gf = self.log_gf[mask_valid]
            self.gf     = self.gf[mask_valid]

            self.damping = self.damping[mask_valid,:]

            self.nu_0_init = self.nu_0_init[mask_valid]

            if rank == 0:
                print(f'\nRemoved {(~mask_valid).sum()} of {len(mask_valid)} transitions with missing damping constants')
            return

        assert(mask_valid.all())

    def oscillator_strength(self, T, Q):

        # [cm^-1 / (atom cm^-2)]
        term1 = (self.gf*np.pi*e**2) / (m_e*c**2)
        term2 = np.exp(-c2*self.E_low/T) / Q
        term3 = (1 - np.exp(-c2*self.nu_0/T))

        S = term1 * term2 * term3 # [cm^1]
        return S
    
    def thermal_broadening(self, T):
        # Gandhi et al. (2020b) [cm^-1]
        self.gamma_G = np.sqrt((2*k_B*T)/self.mass) * self.nu_0/c
        return self.gamma_G
    
    def natural_broadening(self):

        # Gandhi et al. (2020b) [cm^-1]
        self.gamma_N =  0.22 * self.nu_0**2/(4*np.pi*c)

        # Use the provided natural broadening coefficient
        mask_gamma_N = (self.damping[:,0] != 0)
        self.gamma_N[mask_gamma_N] = \
            10**self.damping[mask_gamma_N,0] / (4*np.pi*c)
        return self.gamma_N
    
    def vdW_broadening(
            self, P, T, 
            VMR_H=0, VMR_H2=0.85, VMR_He=0.15, 
            E_H=13.6*8065.73, 
            alpha_H=0.666793, alpha_p=0.806, # H2, Schweitzer et al. (1996)
            mass_p=2.016*amu, # H2
            ):

        # Perturber density (single-assumption)
        #N_p = P*1e6 / (k_B*T) # [cm^-3]

        # Total number density
        N_tot = P*1e6 / (k_B*T) # [cm^-3]
        N_H   = VMR_H * N_tot
        N_H2  = VMR_H2 * N_tot
        N_He  = VMR_He * N_tot

        # Difference in polarizabilities
        C_H, C_H2, C_He = 1, 0.85, 0.42 # Kurucz & Furenlid (1979)

        mask_gamma_vdW = (self.damping[:,2] != 0)

        # Use the provided vdW broadening coefficients (Sharp & Burrows 2007)
        self.gamma_vdW[mask_gamma_vdW] = \
            1/(4*np.pi*c) * 10**self.damping[mask_gamma_vdW,2] * \
            (C_H*N_tot) * (T/10000)**(3/10)
            #(C_H*N_H + C_H2*N_H2 + C_He*N_He) * \

        if self.is_alkali:

            # Schweitzer et al. (1995) [cm^6 s^-1]
            C_6 = 1.01e-32 * alpha_p/alpha_H * (self.Z+1)**2 * \
                ((E_H/(self.E_ion-self.E_low))**2 - (E_H/(self.E_ion-self.E_high))**2)
            C_6 = np.abs(C_6)
            
            self.gamma_vdW = \
                1.664461/(2*c) * (k_B*T * (1/self.mass+1/mass_p))**(3/10) * \
                C_6**(2/5) * N_tot

    def __call__(self, P, T, states, broad_power_law=None, indices_for_power_law=None):

        Q = states.partition_function(T)
        if rank == 0:
            print('Q =', Q)

        # Compute the line strength
        self.S = self.oscillator_strength(T, Q)

        # Retrieve broadening parameters
        self.thermal_broadening(T)
        self.natural_broadening()
        self.vdW_broadening(P, T)

        if (broad_power_law is not None) and (indices_for_power_law is not None):
            
            # Use (P,T)-dependent impact-shifts and widths

            # Update the impact width and shift
            w, d = broad_power_law(P=P, T=T)

            for i, idx in enumerate(indices_for_power_law):
                self.gamma_vdW[idx] = w[i] # Set impact width
                self.nu_0[idx]     += d[i] # Apply impact shift

                if rank == 0:
                    print(1e7/self.nu_0[idx], self.gamma_vdW[idx], self.gamma_N[idx])

        # Lorentz width
        self.gamma_L = self.gamma_vdW + self.gamma_N # [cm^-1]

        self.gamma_L[(self.gamma_L == 0.)] = np.nan
        self.gamma_G[(self.gamma_G == 0.)] = np.nan

        # Voigt width
        self.gamma_V = 0.5436*self.gamma_L + \
            np.sqrt(0.2166*self.gamma_L**2 + self.gamma_G**2)

class BroadeningPowerLaw:
    def __init__(self, A_w, b_w, A_d, b_d, n_ref=1e20):

        # Parameters for impact width
        self.A_w = np.array(A_w)
        self.b_w = np.array(b_w)

        # Parameters for impact shift
        self.A_d = np.array(A_d)
        self.b_d = np.array(b_d)
        
        # Reference density [cm^-3]
        self.n_ref = n_ref
        
    def __call__(self, P, T):

        # Number density [cm^-3]
        n = P*1e6 / (k_B*T)

        # Scale the impact width with (P,T)
        w = self.A_w * T**self.b_w
        w *= n / self.n_ref

        # Scale the impact shift with (P,T)
        d = self.A_d * T**self.b_d
        d *= n / self.n_ref

        return w, d

class CrossSections:
    def __init__(
            self, states_file, transitions_file, pRT_wave_file, 
            mass, E_ion, max_nu_sep=250, max_sep_voigt=False, 
            is_alkali=False, only_trans_with_VALD_damping=True, indices_to_exclude=[], 
            A_w=None, b_w=None, A_d=None, b_d=None, 
            indices_for_power_law=None
            ):
        
        self.broad = None
        self.indices_for_power_law = indices_for_power_law
        if A_w is not None:
            self.broad = BroadeningPowerLaw(
                A_w=A_w, b_w=b_w, A_d=A_d, b_d=b_d
                )

        self.states = States(states_file)
        self.trans  = Transitions(
            transitions_file, mass=mass, E_ion=E_ion, 
            is_alkali=is_alkali, only_trans_with_VALD_damping=only_trans_with_VALD_damping
            )
        
        self.indices_to_exclude = indices_to_exclude
        if self.indices_to_exclude and rank == 0:
            # List not empty
            lines_to_exclude = 1e7/self.trans.nu_0[self.indices_to_exclude]
            print('\nLines to exclude [nm]:')
            print(lines_to_exclude)

        # Maximum wavenumber separation [cm^-1]
        self.max_nu_sep = max_nu_sep
        # Separation as function of Voigt width
        self.max_sep_voigt = max_sep_voigt

        # Wavelength / wavenumber grid
        self.wave = np.loadtxt(pRT_wave_file) * 1e7 # [nm]
        self.nu = 1e7 / self.wave # [cm^-1]

    def line_cutoff_mask(self, nu_0, gamma_V=None):
        
        if gamma_V is not None:
            # Relative to Voigt width
            return np.abs(self.nu-nu_0) < self.max_nu_sep*gamma_V
        
        # Constant cutoff
        return np.abs(self.nu-nu_0) < self.max_nu_sep

    def line_profile(self, nu_0, gamma_L, gamma_G, mask_nu):

        # Gandhi et al. (2020b)
        u = (self.nu[mask_nu] - nu_0) / gamma_G
        a = gamma_L / gamma_G

        f = Faddeeva(u + 1j*a).real / (gamma_G*np.sqrt(np.pi)) # [cm]
        return f
    
    def scaled_line_profile(self, idx):

        if idx in self.indices_to_exclude:
            return 0*self.nu
        
        # Wavenumber of line core
        nu_0_i = self.trans.nu_0[idx]

        # Get line-cutoff mask
        if self.max_sep_voigt:
            mask_nu_i = self.line_cutoff_mask(
                nu_0_i, gamma_V=self.trans.gamma_V[idx]
                )
        else:
            mask_nu_i = self.line_cutoff_mask(nu_0_i)

        # Get the Voigt line-profile
        f_i = np.zeros_like(self.nu)
        f_i[mask_nu_i] = self.line_profile(
            nu_0_i, gamma_L=self.trans.gamma_L[idx], 
            gamma_G=self.trans.gamma_G[idx], mask_nu=mask_nu_i
            )
        
        b_i = 1
        #if self.max_sep_voigt:
        #    b_i = (2/np.pi * np.arctan2(self.max_nu_sep*self.trans.gamma_V[idx], self.trans.gamma_L[idx]))**(-1)
        #else:
        #    b_i = (2/np.pi * np.arctan2(self.max_nu_sep, self.trans.gamma_L[idx]))**(-1)
        #b_i = (2/np.pi * np.arctan2(self.max_nu_sep, self.trans.gamma_L[idx]))**(-1)

        # Scale by the line-strength
        sigma_i = b_i * self.trans.S[idx] * f_i
        return sigma_i
    
    def get_opacity(self, T, P, output_dir='./data/'):

        self.trans.nu_0 = np.copy(self.trans.nu_0_init)

        # Update the line-widths and -strengths
        self.trans(
            P=P, T=T, states=self.states, 
            broad_power_law=self.broad, 
            indices_for_power_law=self.indices_for_power_law
            )

        # Start with zero opacity
        sigma_per_rank = np.zeros_like(self.nu)

        # Parallelise for-loop
        iterable = np.arange(len(self.trans.nu_0))
        
        n_iter = len(iterable)
        n_procs = comm.Get_size()

        # Number of iterables to compute per process
        per_rank = int(n_iter / n_procs) + 1

        # Lower, upper indices to compute for this rank
        low, high = rank*per_rank, (rank+1)*per_rank

        pbar = lambda a: a
        if rank == 0:
            pbar = tqdm
        
        # Run the function
        for i in pbar(range(low, high)):
            if i >= n_iter:
                break
            idx = iterable[i]

            sigma_per_rank += self.scaled_line_profile(idx)

        # Pause until all processes have finished
        comm.Barrier()

        # Sum the outputs of all processes together
        sigma_wave_range = comm.reduce(sigma_per_rank, op=MPI.SUM, root=0)
        
        if rank == 0:
            #sigma = np.ones_like(self.wave)*1e-250
            #sigma[self.mask_wave_range] = sigma_wave_range
            sigma = sigma_wave_range

            np.savetxt(
                '{}/sigma_{:.0f}.K_{:6f}bar.dat'.format(output_dir, T, P), 
                np.column_stack((self.wave*1e-7, sigma))
                )
        
        # Pause until all processes have finished
        comm.Barrier()

if __name__ == '__main__':

    '''
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

        #2000., 
        #2500., 
        #2750., 
        #3250., 
        #3500., 
        #3750., 
        #4000., 
    ])
    #'''
    '''
    temperatures = np.array([
        500, 
        600, 
        725, 
        1000, 
        #1250, 
        1500, 
        2000, 
        2500, 
        3000
    ])
    #'''
    temperatures = np.array([1000, 3000]); pressures = np.array([0.00001, 1.0])
    '''
    pressures = np.array([
        0.000001, 
        0.00001, 
        0.0001, 
        0.001, 
        0.01, 
        0.1, 
        1.0, 
        10.0, 
        100.0, 
        1000.0, 
    ])
    '''

    #'''
    CS = CrossSections(
        states_file='./data/Fe_I_states.txt',              # NIST levels
        transitions_file='./data/Fe_I_transitions.txt',    # VALD transitions
        pRT_wave_file='./data/wlen_petitRADTRANS.dat',     # pRT wavelengths
        mass=55.845*amu,           # Atomic mass
        E_ion=63737.704,              # https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
        is_alkali=False, only_trans_with_VALD_damping=True, # Use only transitions with a valid damping constant
        max_nu_sep=60,              # Line-wing cutoff [cm^-1]
    )
    #'''
    
    '''
    CS = CrossSections(
        states_file='./data/Na_I_states.txt',              # NIST levels
        transitions_file='./data/Na_I_transitions_ck.txt', # Transitions on full pRT wavelength-extent (including c-k)
        pRT_wave_file='./data/wlen_petitRADTRANS.dat',     # pRT wavelengths
        mass=22.989769*amu,           # Atomic mass
        E_ion=41449.451,              # https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
        is_alkali=True,               # Use Schweitzer (1995) equation for gamma_vdW
        max_nu_sep=4500,              # Line-wing cutoff [cm^-1]
        indices_to_exclude=[499,500], # (optical) doublet indices
    )
    #'''

    for T in temperatures:
        for P in pressures:
            if rank == 0:
                print('\nT={:.0f} K | P={:6f} bar'.format(T, P))

            # Give output directory
            CS.get_opacity(T, P, output_dir='/net/lem/data2/regt/Fe_test/')