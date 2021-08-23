import numpy as np

from babyrixs.constants import HBAR
from babyrixs.observables import observable
from babyrixs.pumpedchain import Pumped1Dchain
from babyrixs.structure import Structure
from babyrixs.rk4on import Rk4on

#---------------------------------------------------
# Parameters for bare system (including pump-pulse)
#---------------------------------------------------
# System size
N = 2
# hopping potential
gamma = 1
# Hubbard-potential
U = 2
# Filling
Ne = 2

# Pump-pulse intensity
e0_pump = 0.5
# Pump-pulse frequency
om_pump = 1
# Pump-pulse width
wd_pump = 2
# Center of the pump pulse
t0_pump = 10

#-------------------------------
# Paramaters for time-evolution
#-------------------------------
# Final time of simulation
tf = 60
# Time-step of simulation
dt = 0.05
# Time-vector
times = np.arange(0, tf+dt, dt)

#------------------------------------------------
# Parameters for probe-pulse (including spectra)
#------------------------------------------------
# Width of the probe-pulse
wd_probe = 0.5
# Range of frequencies for spectra
wi = -6
wf = 6
dw = 0.1
omegas = np.arange(wi, wf+dw, dw)
# time-step to obtain spectra
dt_spec = dt

#---------------------
# Set the bare system
#---------------------
bare_system = Pumped1Dchain(N=N,
                            gamma=gamma,
                            U=U,
                            Ne=Ne,
                            e0_pump=e0_pump,
                            om_pump=om_pump,
                            wd_pump=wd_pump,
                            t0_pump=t0_pump)

#-------------------------------------------
# Perform the time-evolution of the system
#-------------------------------------------
# Observable (and the operator associated to QFI)
charge_i = observable(bare_system, 'charge_i=1')
direct_QFI = np.zeros(len(times))
# Time-evolution
system_in_time = Rk4on(bare_system)
for a, t in enumerate(times):
    O_of_t = system_in_time.heiesenberg(charge_i)
    O_square_of_t = np.dot(O_of_t, O_of_t)
    E_of_t = system_in_time.expectation(O_square_of_t, using='gs')
    direct_QFI[a] = E_of_t / 4
    print('Time ::  t = ', t)
    system_in_time.propagate(t, dt)

# Save QFI in file
data = np.zeros((len(times), 2))
data[:, 0] = times
data[:, 1] = direct_QFI
np.savetxt('./results/direct_QFI_tf=60.txt', data)


#----------------------------------------
# Evaluation of structure factor spectra
#----------------------------------------
# Create the object for structure factor calculations
s_factor = Structure(bare_system, charge_i)

# Find the spectra (w = variable, t = variable)
spectra = s_factor.spectra(wi=wi, wf=wf, dw=dw,
                           tf=tf, dt_spec=dt_spec, dt=dt,
                           wd_probe=wd_probe,
                           visual=True, verbose=True,
                           save=True, filename='./results/sneq_wt_tf=60')

