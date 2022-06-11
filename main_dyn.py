import numpy as np

from babyrixs.observables import observable
from babyrixs.pumpedchain import Pumped1Dchain
from babyrixs.rk4on import Rk4on
from babyrixs.constants import HBAR

import matplotlib.pyplot as plt

# import beauty.tanya
# from beauty.morph import cm2in

#------------
# Parameters
#------------
# System size
N = 2
# hopping potential (eV)
gamma = 1
# Hubbard-potential (eV) -- Small value is to remove degeneracy
U = 2
# Filling
Ne = 2

# Pump-pulse intensity
e0_pump = 0.5
# Pump-pulse frequency (eV)
om_pump = 3
# Pump-pulse width (fs)
wd_pump = 2
# Center of the pump pulse
t0_pump = 10

# Time-parameters (fs)
ti = 0
tf = 30
dt = 0.01
time = np.arange(ti, tf+dt, dt)

#-----------------------------------------------
# Time-evolution of the time-evolution operator
#-----------------------------------------------
# Set the bare system
bare_system = Pumped1Dchain(N=N,
                            gamma=gamma,
                            U=U,
                            Ne=Ne,
                            e0_pump=e0_pump,
                            om_pump=om_pump,
                            wd_pump=wd_pump,
                            t0_pump=t0_pump)

# Set the observables you want to track
O1 = observable(bare_system, 'charge_fluctuations')
O2 = observable(bare_system, 'spin_fluctuations')

O1_vec = np.zeros(len(time))
O2_vec = np.zeros(len(time))

# Set the system to evolve in time by passing bare-system to it
system_in_time = Rk4on(bare_system)
for a, t in enumerate(time):
    # Evaluate time-dependent operators in Heiesenberg-Picture
    O1_t = system_in_time.heiesenberg(O1)
    O2_t = system_in_time.heiesenberg(O2)
    # Caluclate their expectation value
    E1_t = system_in_time.expectation(O1_t, using='gs')
    E2_t = system_in_time.expectation(O2_t, using='gs')
    # Load results in to arrays (for plotting)
    O1_vec[a] = E1_t
    O2_vec[a] = E2_t
    # Update the state of the system
    print('Time ::  t = ', t, ' fs')
    system_in_time.propagate(t, dt)

#-------------------
# Plot your results
#-------------------
# For charge fluctuations
fig = plt.figure(figsize=(8.5/2.54, 8.5/2.54))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.plot(time, O1_vec, lw=2, color='C0')
ax.set_xlabel(r'Time t ($\mathrm{\gamma/\hbar}$)')
ax.set_ylabel(r'Charge fluctuations ($\mathrm{e^2}$)')
fig.savefig('./results/charge.pdf', bbox_inches='tight')

# For total spin fluctuations
fig = plt.figure(figsize=(8.5/2.54, 8.5/2.54))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.plot(time, O2_vec, lw=2, color='C0')
ax.set_xlabel(r'Time t ($\mathrm{\gamma/\hbar}$)')
ax.set_ylabel(r'Spin fluctuations ($\mathrm{\hbar^2/4}$)')
fig.savefig('./results/spin.pdf', bbox_inches='tight')

