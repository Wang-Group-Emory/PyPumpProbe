import numpy as np
from scipy import integrate
from scipy.fft import fft, ifft

from babyrixs.fit import Fit


import matplotlib.pyplot as plt

class Analysis:

    # TODO: Can we implement something to find wd_probe directly from the data.
    def __init__(self, datafile, timesfile, omegafile,
                 tcenter=None, wd_pump=None, wd_probe=None, order='wt'):
        """ Class to perform Fourier transform related analysis on
            the structure factor data.

        Parameters:
        -----------
        datafile: string
            The name of the .txt file which contains the S(omega, t) data
            of the structure factor.
        timesfile: string
            The name of the .txt file which contains the times used in the
            spectra generation.
        omegafile: string
            The name of the .txt file which contains the omega values used in
            the spectra generation.
        tcenter: float
            Center of the pump-pulse.
        wd_pump: float
            Width of the pump-pulse.
        wd_probe: float
            The width of the probe-pulse.

        Attributes:
        -----------
        datafile: string
            The name of the .txt file which contains the S(omega, t) data
            of the structure factor.
        timesfile: string
            The name of the .txt file which contains the times used in the
            spectra generation.
        omegafile: string
            The name of the .txt file which contains the omega values used in
            the spectra generation.
        tcenter: float
            Center of the pump-pulse.
        wd_pump: float
            Width of the pump-pulse.
        wd_probe: float
            The width of the probe-pulse.
        order: string
            'wt' means time is on x-axis in the data and 'tw' means it is on
             y-axis.
        files_consistent: Bool
            If True, it means that the files are consistent in terms of the shape
            of the data.
        """
        self.datafile = datafile
        self.timesfile = timesfile
        self.tcenter = tcenter
        self.wd_pump = wd_pump
        self.wd_probe = wd_probe
        self.omegafile = omegafile
        self.order = order
        self.files_consistent = self.check_data_consistency()

        return

    def check_data_consistency(self):
        if self.order == 'wt':
            data = np.loadtxt(self.datafile)
        else:
            data = np.loadtxt(self.datafile).T
        times = np.loadtxt(self.timesfile)
        omega = np.loadtxt(self.omegafile)
        ny, nx = np.shape(data)

        if nx==len(times) and ny==len(omega):
            status = True
        else:
            raise valueError("The datafile for structure factor \
                              does not match the size of time or omega file.")
        return status

    def integrate_w(self, visual=False):
        """ Function to integrate the S(omega, t) data on omega
            and return the final sum as function of time. Note that
            it returns a 1D array which represents the values as a
            function of time once the integration on omega is complete.
        """
        # Obtain the frequency (w) grid
        omega = np.loadtxt(self.omegafile)
        # Extract S(w, t) and integrate over w
        if self.order == 'wt':
            data = np.loadtxt(self.datafile)
        else:
            data = np.loadtxt(self.datafile).T
        data_integrated = integrate.simpson(data, omega, axis=0)

        # Visualize the data if needed
        if visual:
            fig = plt.figure(figsize=(5, 3))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            times = np.loadtxt(self.timesfile)
            ax.plot(times, data_integrated, lw=2, color='C0')
            ax.set_xlabel(r" Time $\mathrm{t'}$ ")
            ax.set_ylabel(r"${\rm S(t') = \int d\omega\, S_{neq}(\omega, t')}$")
            fig.savefig('./results/s_of_t.pdf', bbox_inches='tight')

        return data_integrated

    def find_seq_sneq_and_beta(self, threshold=4, avg_window=1,
                               deviation=1, visual=False):
        """ Function to return the equilibrium value of S(w, t). Here
            threshold is the time from t=0 or t=tf that is neglected,
            and window is the amount of time over which the averaging is
            done to figure out sneq.
        """
        # Obtain s_of_t
        s_of_t = self.integrate_w(visual=visual)

        # Find index at which t = thresold which gives seq
        times = np.loadtxt(self.timesfile)
        pos = np.argmin( np.abs(times - threshold) )
        seq = s_of_t[pos]

        # Find pos_i and pos_f where you take mean near t = tf
        tf = times[-1]
        tavg_f = tf - threshold
        tavg_i = tf - threshold - avg_window
        pos_f = np.argmin( np.abs(times - tavg_f) )
        pos_i = np.argmin( np.abs(times - tavg_i) )
        sneq = np.mean( s_of_t[pos_i:pos_f] )

        # Find the non-trivial window for plotting
        tcenter = self.tcenter
        wd_pump = self.wd_pump
        window_i = tcenter - wd_pump/5
        window_f = tcenter + wd_pump/5
        window = window_f - window_i
        beta = 1 / window

        # Visulaize if needed
        if visual:
            new_f = sneq + (seq - sneq) * self.fermi(times,
                                                     beta=beta,
                                                     t0=tcenter)
            fig = plt.figure(figsize=(5, 3))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(times, s_of_t, lw=2, color='C0', label='S(t)')
            ax.plot(times[::10], new_f[::10], lw=0.1, color='k',
                    marker='.', markersize=2, label='Fermi component')
            ax.axvline(times[pos], lw=0.5, ls=':', color='k')
            ax.axvline(times[pos_i], lw=0.5, ls=':', color='k')
            ax.axvline(times[pos_f], lw=0.5, ls=':', color='k')
            ax.axvline(window_i, lw=0.5, ls=':', color='C3')
            ax.axvline(window_f, lw=0.5, ls=':', color='C3')
            ax.axvline(tcenter - 10*(2*wd_pump/5), lw=2, ls='-', color='C3')
            ax.axvline(tcenter + 10*(2*wd_pump/5), lw=2, ls='-', color='C3',
                       label='Integration window')
            ax.legend(frameon=False)
            ax.set_xlim([0, times[-1]])
            ax.set_xlabel(r" Time $\mathrm{t}$ ")
            ax.set_ylabel(r"${\rm S(t) = \int d\omega\, S_{neq}(\omega, t)}$")
            ax.set_ylim([0.62, 0.82])
            fig.savefig('./results/s_of_t_marked.pdf', bbox_inches='tight')

        return seq, sneq, beta

    def fermi(self, tau, beta=1, t0=10):
        """ Function that returns the Fermi-function
        """
        den = np.exp( beta*(tau - t0) ) + 1
        fermi = 1/den
        return fermi

    def gauss(self, tau, t0=10, sigma=1):
        """ Function that returns the Gaussian function
        """
        norm = 1 / ( sigma * np.sqrt(2*np.pi))
        gs = norm * np.exp( -(tau - t0)**2 / (2*sigma**2) )
        return gs

    def integrate_FermiGauss(self, times, sigma=1, beta=1, t0=10,
                             num_sigmas=10, num_kbTs=10, kbT_steps=50,
                             visual=True):
        """ Function that integrates the product of a Gaussian function
            and fermi-function.

        Parameters:
        -----------
        times: numpy 1D array
            The times at which the integrated result is needed.
        sigma: float
            The width of the probe-pulse.
        beta: float
            The inverse temperature of the Fermi-function.
        t0: float
            The position where the Fermi-function is centered.
        num_sigmas: float
            The number of sigma lengths considered in the integration.
        num_kbTs: float
            The number of kbT lengths considered in the integration.
        kbT_steps: int
            The number of points considered between one kbT
        visual: bool
            Option to visualize results
        Returns:
        --------
        FermiGauss: numpy 1D array
            The final result of integration of the Fermi-function
            and the Gaussian function.
        """
        # Create the integration window
        kbT = 1/beta
        ti = t0 - (num_kbTs*kbT) - (num_sigmas*sigma)
        tf = t0 + (num_kbTs*kbT) + (num_sigmas*sigma)
        dt = kbT / kbT_steps
        time_int = np.arange(ti, tf+dt, dt)

        # Now create the mesh grid on which the function is defined
        t, tau = np.meshgrid(times, time_int)

        # Create Fermi-part
        fermi_part = 1 / ( np.exp( beta*(tau - t0)) + 1)

        # Create the Gaussian part
        gauss_part = np.exp( -(tau - t)**2 / (2*sigma**2) )
        gauss_part *= 1 / (sigma*np.sqrt(2*np.pi))

        # Take the product
        product = fermi_part * gauss_part

        # Perform integration using simpsons method
        FermiGauss = integrate.simpson(product, time_int, axis=0)

        # Visualize if needed
        if visual:
            fig = plt.figure(figsize=(5, 4))
            gs = fig.add_gridspec(2, 1, hspace=0.5)
            ax = fig.add_subplot(gs[0, 0])
            bx = fig.add_subplot(gs[1, 0])

            ax.plot(time_int, fermi_part[:, 0], color='C0', lw=2,
                    label=r'${\rm f(\tau)}$')
            ax.plot(time_int, gauss_part[:, 100], color='C1', lw=2,
                    label=r'${\rm g(t, \tau)}$')
            ax.set_xlabel(r'$\mathrm{\tau}$')
            ax.set_ylabel(r'${\rm f(\tau)~and~g(t, \tau)}$')
            ax.set_xlim([ti, tf])
            ax.legend(frameon=False)

            bx.plot(time_int, fermi_part[:, 0], color='C0', lw=2,
                    label=r'${\rm f(\tau)}$', ls='-')
            bx.plot(times, FermiGauss, color='C3', lw=2)
            bx.set_xlim([times[0], times[-1]])
            bx.set_xlabel('t')
            bx.set_ylabel(r'${\rm \int d\tau g(t, \tau) f(\tau)}$')

            fig.savefig('./results/fermigauss.pdf', bbox_inches='tight')

        return FermiGauss


    def fit_QFI(self, N=10,
                      threshold=4,
                      avg_window=1,
                      deviation=1,
                      visual=False):
        """ Function to return the quantum Fisher information from
            the provided spectra.
        """
        # First get Seq, Sneq, beta
        seq, sneq, beta = self.find_seq_sneq_and_beta(threshold=threshold,
                              avg_window=avg_window,
                              deviation=deviation,
                              visual=visual)

        #-------------------------------------
        # Construction of the R_of_t function
        #-------------------------------------
        # Find index at which t = thresold which serves as starting point for fit
        times = np.loadtxt(self.timesfile)
        pos_i = np.argmin( np.abs(times - threshold) )

        # Find pos_f that serves at the ending point for the fit
        tf = times[-1]
        tavg_f = tf - threshold
        pos_f = np.argmin( np.abs(times - tavg_f) )

        # Isolate the corressponding time and the data
        times_iso = times[pos_i:pos_f]
        s_of_t_iso = self.integrate_w(visual=visual)[pos_i:pos_f]

        # Generate the FermiGauss part of the function
        sigma = self.wd_probe / np.sqrt(2)
        FermiGauss = self.integrate_FermiGauss(times_iso,
                                          sigma=sigma,
                                          beta=beta,
                                          t0=self.tcenter,
                                          visual=visual)
        # R_of_t is constructed
        R_of_t = (s_of_t_iso
                  - sneq
                  - (seq - sneq)*FermiGauss )

        # Use R_of_t to find A_i, P_i, and W_i
        make_fit = Fit(R_of_t, times_iso)
        A_i, P_i, W_i = make_fit.give_Ai(N=N, sigma=sigma, visual=visual)

        # Construct QFI
        sum_of_gaussians = np.zeros(len(times_iso))
        for a in range(N):
            gauss = A_i[a] * np.exp(-(times_iso - P_i[a])**2 / (2*W_i[a]**2))
            sum_of_gaussians += gauss

        qfi_iso = sneq + (seq - sneq)*FermiGauss + sum_of_gaussians
        qfi_iso *= (8 * self.wd_probe * np.sqrt(np.pi) )

        return times_iso, qfi_iso
