import numpy as np

from analysis import Analysis
from scipy.fft import fft

import matplotlib.pyplot as plt
import beauty.tanya
from beauty.morph import cm2in

def gauss(t, t0, sigma):
    num = np.exp( -(t - t0)**2 / (2*sigma**2) )
    den = sigma * np.sqrt(2*np.pi)
    res = num / den
    return res

#------------
# Parameters
#------------
t0 = 0
sigma_t = 0.5
sigma_w = 1

#---------------------
# Data-size variables
#---------------------
N = 200
M = 2*N + 1

#-------------------------
# Times and omega vectors
#-------------------------
times = np.linspace(-10, 10, M)
omega = np.linspace(-10, 10, M)

#-----------------------------------------
# Construct a w and t dependent function
#-----------------------------------------
# t-dependent part
ft = gauss(times, t0, sigma_t)
# w-dependent part
gw = gauss(omega, 0, sigma_w)
# w and t -dependent function
swt = np.zeros((len(omega), len(times)))
for a, w in enumerate(omega):
    for b, t in enumerate(times):
        swt[a, b] = gw[a]*ft[b]

# Generate the required files for testing
datafile = './results/sneq_wt.txt'
timesfile = './results/sneq_wt_times.txt'
omegafile = './results/sneq_wt_omega.txt'
np.savetxt(datafile, swt)
np.savetxt(timesfile, times)
np.savetxt(omegafile, omega)

#----------------------------------------------------------------
# Calculate the result of integration of S(w, t) on frequency w,
# followed by FFT on time t, to obtain S as  function of w_prime.
# Here we use the 'analysis' module.
#----------------------------------------------------------------
sneq_wt  = Analysis(datafile,
                    timesfile,
                    omegafile)
sneq_t = sneq_wt.integrate_w()
w_prime, sneq_w_prime = sneq_wt.integrate_w_and_fft_t()

#------------------------------------
# Analytical result in w_prime space
#------------------------------------
om_an = np.linspace(-10, 10, M)
fom_an = (np.sqrt(2*np.pi) / sigma_t) * gauss(om_an, 0, 1/sigma_t)

#------------------
# Plot the results
#------------------
fig = plt.figure( figsize=(cm2in(12), cm2in(5)) )
gs = fig.add_gridspec(1, 2, wspace=0.5)
ax = fig.add_subplot(gs[0, 0])
bx = fig.add_subplot(gs[0, 1])
ax.tick_params(which='both', direction='in', size=2)
bx.tick_params(which='both', direction='in', size=2)

ax.plot(times, sneq_t, color='C0', lw=2)
ax.set_xlabel('Time t')
string = r'${\rm S(t) = \int S_\mathrm{neq}(\omega, t)\, d\omega}$'
ax.set_ylabel(string)
ax.set_xlim([-10, 10])

bx.plot(om_an, fom_an, lw=2, color='C1', label='analytical')
bx.plot(w_prime, np.real(sneq_w_prime),
        lw=0.2, color='k', ls='-', marker='.', markersize=1, label='babyrixs')
bx.legend(frameon=False)
bx.set_xlim([-10, 10])
bx.set_ylim([-0.1, 2])
bx.set_xlabel(r"Frequency $\mathrm{\omega'}$")
bx.set_ylabel(r"${\rm \int S(t)\exp(i\omega' t)\, dt}$")

#---------------
# Annotations
#---------------
pos_x = 0.5
pos_y = 1.2
string = r'${\rm S_\mathrm{neq}(\omega, t) \equiv g(\omega, \sigma_\omega) g(t, \sigma_t)}$'
ax.text(pos_x, pos_y, string, transform=ax.transAxes)

pos_x = 0.05
pos_y = 0.9
ax.text(pos_x, pos_y, r'${\rm \sigma_\omega = 1}$', transform=ax.transAxes)

pos_x = 0.05
pos_y = 0.8
ax.text(pos_x, pos_y, r'${\rm \sigma_t = 0.5}$', transform=ax.transAxes)

#-------------
# Save figure
#-------------
fig.savefig('./results/test_analysis.pdf', bbox_inches='tight')



