import numpy as np

from constants import HBAR
from observables import observable
from pumpedchain import Pumped1Dchain
from structure import Structure

#----------------------------
# Parameters for bare system
#----------------------------
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
om_pump = 3.1
# Pump-pulse width (fs)
wd_pump = 2
# Center of the pump pulse
t0_pump = 10

# Set the bare system
bare_system = Pumped1Dchain(N=N,
                            gamma=gamma,
                            U=U,
                            Ne=Ne,
                            e0_pump=e0_pump,
                            om_pump=om_pump,
                            wd_pump=wd_pump,
                            t0_pump=t0_pump)

#-------------------------------------------------
# Parameters  for the structure factor calculation
#-------------------------------------------------
# Final time of simulation
tf = 20
# Time-step of simulation
dt = 0.1
# Frequency of probe-pulse
om_probe = 1
# Width of the probe-pulse
wd_probe = 0.5
# Time at which you may need spectrum or spectra
t = 3
# If doing range of frequencies
wi = -6
wf = 10
dw = 0.2
# time-step to obtain spectra
dt_spec = 0.5

#----------------------------------------
# Evaluation of structure factor spectra
#----------------------------------------
# Operator associated to structure factor
charge_i = observable(bare_system, 'charge_i=1')

# Create the object for structure factor calculations
s_factor = Structure(bare_system, charge_i)



# Find <O(t1)O(t2)>
correlation = s_factor.correlation(tf=tf, dt=dt, visual=True)

# Find the specturm (w = fixed, t = fixed)
spectrum = s_factor.spectrum(om_probe=om_probe, t=t,
                             wd_probe=wd_probe,
                             tf=tf, dt=dt, visual=True)

# Find the spectra (w = variable, t = fixed)
spectra = s_factor.spectra_w(wi=wi, wf=wf, dw=dw, t=t,
                             tf=tf, dt=dt,
                             visual=True, verbose=True)

# Find the spectra (w = variable, t = variable)
spectra = s_factor.spectra(wi=wi, wf=wf, dw=dw,
                           tf=tf, dt_spec=dt_spec, dt=dt,
                           wd_probe=wd_probe,
                           visual=True, verbose=True)

# Find the equilibrium spectra (w = variable)
spectra = s_factor.spectra_eq(wi=wi, wf=wf, dw=dw, visual=True)

