import numpy as np
from babyrixs.analysis import Analysis

import matplotlib.pyplot as plt
import beauty.tanya
from beauty.morph import cm2in

# Pump and Probe pulse info
tcenter = 10
wd_pump = 2
wd_probe = 0.5

# File names of the data
datafile = './results/sneq_wt_tf=60.txt'
timesfile = './results/sneq_wt_tf=60_times.txt'
omegafile = './results/sneq_wt_tf=60_omega.txt'

# Perform analysis
sneq_wt = Analysis(datafile,
                   timesfile,
                   omegafile,
                   tcenter=tcenter,
                   wd_pump=wd_pump,
                   wd_probe=wd_probe,
                   order='wt')

t_qfi, qfi = sneq_wt.fit_QFI(N=10, visual=True)

# Get QFI directy from the Wave-function
data = np.loadtxt('./results/direct_QFI_tf=60.txt')
t_dir = data[:, 0]
qfi_dir = data[:, 1]

# Plot results
fig = plt.figure(figsize=(4, 2.5))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0, 0])

ax.plot(t_dir, 16*qfi_dir, lw=1, color='k',
        label=r'from WF')
ax.plot(t_qfi, qfi, ls='--', color='C2', label='babyrixs', lw=1)
ax.legend(frameon=False)
ax.set_xlabel(r' Time t (${\rm \hbar/\gamma}$)')
ax.set_ylabel(r' ${\rm F_\mathrm{Q}(t)}$')
fig.savefig('./results/qfi_check.pdf', bbox_inches='tight')

