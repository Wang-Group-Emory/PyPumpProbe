import numpy as np

from rk4on import Rk4on

import matplotlib.pyplot as plt
from scipy.linalg import eigh

class Structure:

    def __init__(self, bare_system, operator):
        """ Class for calculating the structure factor assocuated
            to an operator.

        Parameters:
        ----------
        bare_system: object of Pumped1Dchain class
            This allows our class to know what system we are interested in
        operator: numpy 2D array
            This is the matrix form of the operator whose structure factor
            we want to calculate.

        Attributes:
        -----------
        Same as above.
        """
        self.bare_system = bare_system
        self.operator = operator

        return

    def gauss(self, t, tau, sigma):
        """ Simple Gaussian function
        """
        num = np.exp( -(t - tau)**2 / (2*sigma**2) )
        den = np.sqrt(2*np.pi) * sigma
        value = num/den
        return value

    # TODO: We can parallelize the for loops that calculate operator avg.
    # i.e., the ones on t1 and t2. Multiprocessing?
    def correlation(self, tf=5, dt=0.1, visual=False, cmap='bwr'):
        """ Function to calculate the correlation function < O(t1)O(t2) >
            in the integrand of structure factor which is a function of
            t1 and t2. It is evaluated on a grid of t1 and t2 values
            that range from 0 to tf in steps of dt.

        Parameters:
        -----------
        tf: float
            Final time of the simulation.
        dt: float
            Time step of the simulation.
        visual: Bool
            Boolean to decide to 2Dplot the correlation function.
        cmap: string
            Decides the colormap of the 2Dplot.

        Returns:
        --------
        correlation_values: numpy 2D array (dcomplex)
            Returns a complex matrix that contains < O(t1)O(t2) >.

        """
        # Find system size
        N = self.bare_system.N
        dim = 4**N

        # Set the system to evolve in time, and save operator
        # as a function of time.
        time = np.arange(0, tf+dt, dt)
        opr = np.zeros((dim, dim, len(time)), dtype='complex')
        system = self.bare_system
        system_in_time = Rk4on(system)
        for a, t_val in enumerate(time):
            O_t = system_in_time.heiesenberg(self.operator)
            opr[:, :, a] = O_t
            system_in_time.propagate(t_val, dt)

        # Now calculate the integrand on a grid
        psiE0 = system.psiE0
        correlation_values = np.zeros((len(time), len(time)), dtype='complex')
        for a, t1 in enumerate(time):
            for b, t2 in enumerate(time):
                o1 = opr[:, :, a]
                o2 = opr[:, :, b]
                o1o2 = np.dot(o1, o2)
                avg_o1o2 = np.dot(o1o2, psiE0)
                avg_o1o2 = np.dot(np.conjugate(psiE0), avg_o1o2)
                correlation_values[b, a] = avg_o1o2

        # Visualize it
        if visual:
            vmax = np.max(np.max(np.abs(correlation_values)))
            fig = plt.figure(figsize=(6, 5))
            gs = fig.add_gridspec(1, 2, wspace=0.5)
            ax = fig.add_subplot(gs[0, 0])
            bx = fig.add_subplot(gs[0, 1])
            ax.imshow(np.real(correlation_values),
                      origin='lower', cmap=cmap, aspect='equal',
                      extent=(0, tf, 0, tf), vmax=vmax, vmin=-vmax)
            ax.set_xlabel('$\mathrm{t_1}$ ($\mathrm{\hbar/\gamma}$)')
            ax.set_ylabel('$\mathrm{t_2}$ ($\mathrm{\hbar/\gamma}$)')
            bx.imshow(np.imag(correlation_values),
                      origin='lower', cmap=cmap, aspect='equal',
                      extent=(0, tf, 0, tf), vmax=vmax, vmin=-vmax)
            bx.set_xlabel('$\mathrm{t_1}$ ($\mathrm{\hbar/\gamma}$)')
            bx.set_ylabel('$\mathrm{t_2}$ ($\mathrm{\hbar/\gamma}$)')

            ax.title.set_text(r'$\mathrm{Re\langle\hat{O}(t_1)\hat{O}(t_2)\rangle}$')
            bx.title.set_text(r'$\mathrm{Im\langle\hat{O}(t_1)\hat{O}(t_2)\rangle}$')
            fig.savefig('./results/correlation.pdf', bbox_inches='tight')

        return correlation_values


    def spectrum(self, om_probe=1, t=1, wd_probe=0.5,
                 tf=5, dt=0.1, wd_cutoff=5, visual=False, cmap='bwr'):
        """ Function to calculate the spectrum for a single
            value of frequency of the probe-pulse and a single
            value of time t.

        Parameters:
        -----------
        om_probe: float
            The frequency of the probe-pulse.
        t: float
            The time at which you need the spectrum.
        wd_probe: float
            The width of the probe pulse.
        tf: float
            The final time of the simulation.
        dt: float
            The time-step of the simulation.
        wd_cutoff: float
            Decides the box of integration around
            t1 = t2 = t in the t1-t2 plane.
        visual: Bool
            Decide whether to visualize the box of integration around
            t1 = t2 = t in the t1-t2 plane.
        cmap: string
            Colormap of the 2Dplot.

        Returns:
        --------
        spectrum_value: float
            The value of the structure factor at the given frequency
            and time.

        """
        # Obtain the FULL correlation matrix
        corr_full = self.correlation(tf=tf, dt=dt, visual=True)

        # Find the spectrum and visualization information
        t_min, t_max, \
        n_min, n_max, \
        spectrum_value = self.spectrum_box(corr_full,
                                           om_probe=om_probe,
                                           t=t,
                                           wd_probe=wd_probe,
                                           wd_cutoff=wd_cutoff,
                                           tf=tf, dt=dt)

        # Plot for visualization (if needed)
        if visual:
            time = np.arange(0, tf+dt, dt)
            for a, t1 in enumerate(time):
                for b, t2 in enumerate(time):
                    g1 = self.gauss(t, t1, wd_probe)
                    g2 = self.gauss(t, t2, wd_probe)
                    corr_full[b, a] *= (g1 * g2) * np.exp(1j*om_probe*(t1 - t2))

            vmax = np.max(np.max(np.abs(corr_full)))
            fig = plt.figure(figsize=(6, 5))
            gs = fig.add_gridspec(1, 2, wspace=0.5)
            ax = fig.add_subplot(gs[0, 0])
            bx = fig.add_subplot(gs[0, 1])
            ax.imshow(np.real(corr_full),
                      origin='lower', cmap=cmap, aspect='equal',
                      extent=(0, tf, 0, tf), vmin=-vmax, vmax=vmax)
            self.put_box(ax, t_min, t_max)
            ax.set_xlabel('$\mathrm{t_1}$ ($\mathrm{\hbar/\gamma}$)')
            ax.set_ylabel('$\mathrm{t_2}$ ($\mathrm{\hbar/\gamma}$)')
            bx.imshow(np.imag(corr_full),
                      origin='lower', cmap=cmap, aspect='equal',
                      extent=(0, tf, 0, tf), vmin=-vmax, vmax=vmax)
            self.put_box(bx, t_min, t_max)
            bx.set_xlabel('$\mathrm{t_1}$ ($\mathrm{\hbar/\gamma}$)')
            bx.set_ylabel('$\mathrm{t_2}$ ($\mathrm{\hbar/\gamma}$)')

            ax.title.set_text(r'$\mathrm{Re [g_1g_2\langle\hat{O}(t_1)\hat{O}(t_2)\rangle e^{i\omega(t_1 - t_2)}]}$')
            bx.title.set_text(r'$\mathrm{Im [g_1g_2\langle\hat{O}(t_1)\hat{O}(t_2)\rangle e^{i\omega(t_1 - t_2)}]}$')
            fig.savefig('./results/boxed_correlation.pdf', bbox_inches='tight')

        return spectrum_value

    def spectrum_box(self, corr_full, om_probe=1, t=1,
                     wd_probe=0.5, wd_cutoff=5,  tf=5, dt=0.1):
        """ Function to evaluate the spectrum within the box of integration
            region and also provide information about the box for visualization.

        Parameters:
        -----------
        corr_full: numpy 2D array (complex)
            Full correlation matrix in the t1-t2 plane
        om_probe: float
            Frequency of the probe-pulse for spectrum.
        t: float
            Time at which the spectrum is needed.
        wd_probe: float
            Width of the probe-pulse
        wd_cutoff: float
            Time cutoff window which sets the box of integration.
        tf: float
            Final time of simulation.
        dt: float
            Time step of the simulation.

        Returns:
        --------
        t_min: float
            Box of integration starting time.
        t_max: float
            Box of integration ending time.
        n_min: int
            Box of integration starting index in corr_full matrix.
        n_max: int
            Box of integration ending index in corr_full matrix.
        spectrum_value: float
            The magnitude of the spectrum.

        """
        # Decay time-width of the probe-pulse
        T = wd_cutoff * wd_probe
        # Minimum-time and Maximum times that can be incorporated
        Tmin = T
        Tmax = tf - T
        # Find the region of integration in t1-t2 plane
        if t < Tmin:
            t_min = 0
            t_max = t + T
        elif t > Tmax:
            t_min = t - T
            t_max = tf
        else:
            t_min = t - T
            t_max = t + T
        time = np.arange(0, tf+dt, dt)
        min_array = np.abs(time - t_min)
        max_array = np.abs(time - t_max)
        n_min = min_array.argmin()
        n_max = max_array.argmin()

        # Evaluate the spectra in the selected t1-t2 plane
        scan = np.arange(n_min, n_max+1)
        spectrum_value = 0
        for a, n1 in enumerate(scan):
            for b, n2 in enumerate(scan):
                t1 = time[n1]
                t2 = time[n2]
                g1 = self.gauss(t, t1, wd_probe)
                g2 = self.gauss(t, t2, wd_probe)
                pre_factor = (g1*g2) * np.exp(1j*om_probe*(t1 - t2))
                spectrum_value += pre_factor * corr_full[n2, n1]

        spectrum_value *=  (dt**2) / (2*np.pi)
        spectrum_value = np.real(spectrum_value)

        return t_min, t_max, n_min, n_max, spectrum_value

    def spectra(self, wi=-3, wf=3, dw=0.5, tf=5, dt_spec=0.1, dt=0.1,
                wd_probe=0.5, wd_cutoff=5, verbose=False, visual=False, cmap='magma',
                save=False, filename='sneq_wt'):
        """ Function to evaluate the structure factor spectra for provided
            frequency range and time range.

        Parameters:
        -----------
        wi: float
            Initial value of frequency of probe-pulse.
        wf: float
            Final value of frequency of probe-pulse.
        dw: float
            Steps for frequency of probe-pulse.
        tf: float
            Final time of simulation.
        dt_spec: float
            Time-step for the spectra output.
        dt: float
            Time-step of the simulation.
        wd_probe: float
            Width of the probe-pulse
        wd_cutoff: float
            Time cutoff window which sets the box of integration.
        verbose: Bool
            Print the status of job simulation.
        visual: Bool
            Visualize the 2D spectra after it is evaluated.
        cmap: string
            Colormap used in 2D spectra.
        save: Bool
            Save the result into a text file.
        filename: string
            Filename in which the result is being stored.

        Returns:
        --------
        spectra: numpy 2D array
            The 2D spectra.
        """

        # Obtain the FULL correlation matrix
        corr_full = self.correlation(tf=tf, dt=dt)

        # Loops to calculate our spectra
        omega = np.arange(wi, wf+dw, dw)
        times = np.arange(0, tf+dt_spec, dt_spec)
        spectra = np.zeros((len(omega), len(times)))
        for a, om_probe in enumerate(omega):
            if verbose:
                print('Calculating spectra for w = ', om_probe)
            for b, t in enumerate(times):
                t_min, t_max, \
                n_min, n_max, \
                spectrum_value = self.spectrum_box(corr_full,
                                                  om_probe=om_probe,
                                                  t=t,
                                                  wd_probe=wd_probe,
                                                  wd_cutoff=wd_cutoff,
                                                  tf=tf, dt=dt)
                spectra[a, b] = spectrum_value

        # Visualize if needed
        if visual:
            fig = plt.figure(figsize=(3, 1.5))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(spectra,
                      origin='lower', cmap=cmap, aspect='auto',
                      extent=(0, tf, wi, wf), interpolation='spline16')
            ax.set_xlabel(r'Time t ($\mathrm{\hbar/\gamma}$)')
            ax.set_ylabel(r'$\mathrm{\omega}$ ($\mathrm{\gamma/\hbar}$)')
            ax.title.set_text(r'$\mathrm{S_{neq}(\omega, t)}$')
            fig.savefig('./results/sneq_wt.pdf', bbox_inches='tight')

        # Save the file if chosen to
        if save:
            datafile = filename + '.txt'
            timesfile = filename + '_times.txt'
            omegafile = filename + '_omega.txt'
            np.savetxt(datafile, spectra)
            np.savetxt(timesfile, times)
            np.savetxt(omegafile, omega)

        return spectra

    def spectra_w(self, wi=-3, wf=3, dw=0.5, t=2.5, tf=5, dt=0.1,
                  wd_probe=0.5, wd_cutoff=5, verbose=False, visual=False,
                  save=False, filename='sneq_wt.txt'):
        """ Function to evaluate the structure factor spectra for provided
            frequency range at a fixed time.

        Parameters:
        -----------
        wi: float
            Initial value of frequency of probe-pulse.
        wf: float
            Final value of frequency of probe-pulse.
        dw: float
            Steps for frequency of probe-pulse.
        t: float
            Time at which the spectra is desired.
        tf: float
            Final time of simulation.
        dt: float
            Time-step of the simulation.
        wd_pulse: float
            Width of the probe-pulse
        wd_cutoff: float
            Time cutoff window which sets the box of integration.
        verbose: Bool
            Print the status of job simulation.
        visual: Bool
            Visualize the 1D spectra after it is evaluated.

        Returns:
        --------
        spectra: numpy 1D array
            The 1D spectra as a function of frequency.
        """

        # Obtain the FULL correlation matrix
        corr_full = self.correlation(tf=tf, dt=dt)

        # Loops to calculate our spectra
        omega = np.arange(wi, wf+dw, dw)
        spectra = np.zeros(len(omega))
        for a, om_probe in enumerate(omega):
            if verbose:
                print('Calculating spectra for w = ', om_probe)
            t_min, t_max, \
            n_min, n_max, \
            spectrum_value = self.spectrum_box(corr_full,
                                               om_probe=om_probe,
                                               t=t,
                                               wd_probe=wd_probe,
                                               wd_cutoff=wd_cutoff,
                                               tf=tf, dt=dt)
            spectra[a] = spectrum_value

        # Visualize if needed
        if visual:
            fig = plt.figure(figsize=(3, 3))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(omega, spectra, lw=2, color='C3')
            ax.set_xlabel(r'$\mathrm{\omega}$ ($\mathrm{\gamma/\hbar}$)')
            ax.set_ylabel(r'$\mathrm{S_{neq}(\omega, t)}$')
            pos_x = 0.02
            pos_y = 1.05
            string = r' t = ' + str(t) + '$\mathrm{\hbar/\gamma}$'
            ax.text(pos_x, pos_y, string, transform=ax.transAxes)
            fig.savefig('./results/sneq_w.pdf', bbox_inches='tight')

        return spectra

    #TODO: Add temperature to this
    def spectra_eq(self, wi=-3, wf=3, dw=0.1,
                   wd_probe=0.5, temp=0.0, visual=False):
        """ Function to calculate the equilibrium structure factor
            associated to the operator. It uses an analytical
            formula for the calculation.

        Parameters:
        -----------
        wi: float
            Initial value of frequency of probe-pulse.
        wf: float
            Final value of frequency of probe-pulse.
        dw: float
            Steps for frequency of probe-pulse.
        wd_probe: float
            Width of the probe-pulse.
        temp: float
            temperature of the system

        Returns:
        --------
        eqb_spectra: numpy 1D array
            Equilibrium spectra for the provided frequencies.

        """
        # Get all the eigen states
        filling = self.bare_system.Ne
        E0, psi_E0 = self.bare_system.get_gs(filling=filling)
        evals, evecs = eigh(self.bare_system.get_ham(t=-50000))
        # Evaluate the spectra
        omega = np.arange(wi, wf+dw, dw)
        eqb_spectra = np.zeros(len(omega))
        for a, om in enumerate(omega):
            for n, En in enumerate(evals):
                delta_om = En - E0
                sandwich = np.dot(self.operator, evecs[:, n])
                sandwich = np.dot(np.conjugate(psi_E0), sandwich)
                sandwich = np.conjugate(sandwich)*sandwich
                pre_factor = np.real(sandwich)
                gauss = np.exp( -(wd_probe**2) * (om - delta_om)**2 )
                pi_factor = 1/(2*np.pi)
                eqb_spectra[a] += pi_factor * pre_factor * gauss

        # Visualize if needed
        if visual:
            fig = plt.figure(figsize=(3, 3))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(omega, eqb_spectra, lw=2, color='C3')
            ax.set_xlabel(r'$\mathrm{\omega}$ ($\mathrm{\gamma/\hbar}$)')
            ax.set_ylabel(r'$\mathrm{S_\mathrm{eq}(\omega)}$')
            fig.savefig('./results/seq_w.pdf', bbox_inches='tight')

        return eqb_spectra

    def put_box(self, ax, tmin, tmax):
        """ Function to put a box around the integration
            region for g1*g2*<O(t1)O(t2)> integrand.
        """
        x1 = np.arange(tmin, tmax, 0.01)
        y1 = tmin*np.ones(len(x1))
        x2 = tmax*np.ones(len(x1))
        y2 = np.arange(tmin, tmax, 0.01)
        x3 = np.arange(tmin, tmax, 0.01)
        y3 = tmax*np.ones(len(x1))
        x4 = tmin*np.ones(len(x1))
        y4 = np.arange(tmin, tmax, 0.01)
        ax.plot(x1, y1, color='k', lw=1)
        ax.plot(x2, y2, color='k', lw=1)
        ax.plot(x3, y3, color='k', lw=1)
        ax.plot(x4, y4, color='k', lw=1)
        return
