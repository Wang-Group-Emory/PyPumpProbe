import numpy as np

from scipy import integrate
from scipy.fft import fft, ifft

import matplotlib.pyplot as plt

class Analysis:

    # TODO: Can we implement something to find wd_probe directly from the data.
    def __init__(self, datafile, timesfile, omegafile,
                 wd_probe=None, order='wt', cutoff=2.7,
                 dataorigin='sqst', inverselife=3):
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
        wd_probe: float
            The width of the probe-pulse.
        order: string
            Inform what the order of data. 'wt' means that the matrix of spectra
            contains frequency 'w' on the first index of the matrix, and time 't'
            is kept on the second index. 'tw' means that it is other way around.
        cutoff: float
            This is a number that is used to control the high frequnecy cutoff
            in the frequncy domain, beyond which the inverted Gaussian can
            produce incorrect results.
        dataorigin: float
            This option allows babyrixs to identify the origin of the data.
            Right now it supports two options.
            1.) 'sqwt': This means that the data is a structure factor calculation.
            2.) 'rixs': This means that the data is extracted from trRIXS calculation.
        inverselife: float
            This is the inverse-lifetime of the core-hole lifetime and is important
            if the RIXS data is being used to calculate the QFI.

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
        wd_probe: float
            The width of the probe-pulse.
        files_consistent: Bool
            If True, it means that the files are consistent in terms of the shape
            of the data.
        """
        self.datafile = datafile
        self.timesfile = timesfile
        self.omegafile = omegafile
        self.wd_probe = wd_probe
        self.order = order
        self.files_consistent = self.check_data_consistency()
        self.cutoff = cutoff
        self.dataorigin = dataorigin
        self.inverselife = inverselife

        return


    def check_data_consistency(self):
        if self.order=='wt':
            data = np.loadtxt(self.datafile)
        elif self.order=='tw':
            data = np.loadtxt(self.datafile).T
        else:
            raise valueError('Unknown order chosen for the spectra data file')

        times = np.loadtxt(self.timesfile)
        omega = np.loadtxt(self.omegafile)
        ny, nx = np.shape(data)

        if nx==len(times) and ny==len(omega):
            status = True
        else:
            raise valueError("The datafile for structure factor does not match the size of time or omega file.")

        return status


    def k_function(self, Fk, N):
        """ Function to process Fk (discrete fourier transform) and return
            the continuous fourier transform fw by appropriate prefactor
            multiplication.
        """
        k = np.arange(-N, N+1)
        ark = -1j * np.pi * k * ( 1 - 0.5/N )
        prefactor = np.exp(ark)
        fw = prefactor * Fk
        return fw


    def k_function_ift(self, Fk, N):
        """ Function to process Fk (discrete fourier transform) and return
            the continuous fourier transform fw by appropriate prefactor
            multiplication (note this has a + sign compared to the function
            just above).
        """
        k = np.arange(-N, N+1)
        ark = 1j * np.pi * k * (1 - 0.5/N)
        prefactor = np.exp(ark)
        fw = prefactor * Fk
        return fw


    def fast_ft(self, t, ft, visual=False):
        """ Function to perform Fourier transfrom of data in time-domain
            to the frequency-domain as well as the omega values.
        """
        # Check the length of the data and adjust it.
        M = len(ft)
        if np.mod(M, 2)==0:
            # If M is even, we make it odd by repeating the last
            # element one more time.
            print('IT is even ...')
            ft_new = np.zeros(M+1, dtype='complex')
            t_new = np.zeros(M+1)
            ft_new[0:M] = ft
            ft_new[M] = ft[-1]
            ft = ft_new
            t_new[0:M] = t
            t_new[M] = t[-1]
            t = t_new
            M = M + 1

        # Find the N value such that M = 2N + 1
        N = int( (M - 1) / 2 )

        # Perform the FFT operation on ft
        Fk = fft(ft)

        # Extract the left, center, and right values from Fk
        # while fliping the order of left and right values.
        left_values = np.flip(Fk[1:N+1])
        centr_value = Fk[0]
        rite_values = np.flip(Fk[N+1:])

        # Update Fk to get the correct order as -k to k
        Fk[0:N] = left_values
        Fk[N] = centr_value
        Fk[N+1:] = rite_values

        # Obtain fw from Fk by processing with k_function
        fw = self.k_function(Fk, N)

        # Get the list of times to extract dt
        times = t
        dt = times[1] - times[0]

        # Update fw with proper normalization factors
        fw = fw * dt

        # Present the corressponding omega values
        w = np.arange(-N, N+1) * ( np.pi / (N*dt) )

        # Visualize if needed
        if visual:
            fig = plt.figure(figsize=(5, 3))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(w, np.real(fw))
            ax.set_xlabel(r"${\rm \omega' }$")
            ax.set_ylabel(r"${\rm F(\omega') = \int dt' S(t') e^{i\omega' t'} }$")
            fig.savefig('./results/s_of_t_fft.pdf', bbox_inches='tight')

        return w, fw


    def fast_ift(self, t, Fk):
        """ Function to perform inverse Fourier transformation of data in
            frequency domain to the data in time-domain and also return the
            corressponding time values.
        """
        # Check the length of the data and adjust it.
        M = len(Fk)
        if np.mod(M, 2)==0:
            # If M is even, we make it odd by repeating the last
            # element one more time.
            Fk_new = np.zeros(M+1, dtype='complex')
            t_new = np.zeros(M+1)
            Fk_new[0:M] = Fk
            Fk_new[M] = Fk[-1]
            Fk = Fk_new
            t_new[0:M] = t
            t_new[M] = t[-1]
            t = t_new
            M = M + 1

        # Find the N value such that M = 2N + 1
        N = int( (M - 1) / 2 )

        # Use scipt ifft to get the inverse fourier transform
        fm = ifft(Fk)

        # Extract the left, center, and right values from fm
        # and re-order them correctly
        left_values = np.flip(fm[1:N+1])
        centr_value = fm[0]
        rite_values = np.flip(fm[N+1:])
        fm[0:N] = left_values
        fm[N] = centr_value
        fm[N+1:] = rite_values

        # Use DFT result to get contnuous FT result
        fm = self.k_function_ift(fm, N)

        # Obtain the dt value
        times = t
        dt = times[1] - times[0]

        # Divide fm  by dt to get the final result
        ftn = fm / (dt)

        # Construct the time values
        tn = t

        return tn, ftn


    def integrate_w(self, visual=False):
        """ Function to integrate the S(omega, t) data on omega
            and return the final sum as function of time. Note that
            it returns a 1D array which represents the values as a
            function of time once the integration on omega is complete.
        """
        # Find domega
        omega = np.loadtxt(self.omegafile)

        # Extract data and integrate
        if self.order=='tw':
            data = np.loadtxt(self.datafile).T
        else:
            data = np.loadtxt(self.datafile)

        data_integrated = integrate.simpson(data, omega, axis=0)

        # Visualize if needed
        if visual:
            fig = plt.figure(figsize=(5, 3))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            times = np.loadtxt(self.timesfile)
            ax.plot(times, data_integrated)
            ax.set_xlabel(' Time t')
            ax.set_ylabel(r'${\rm S(t) = \int d\omega S(\omega, t)}$')
            fig.savefig('./results/s_of_t.pdf', bbox_inches='tight')

        return data_integrated

    def integrate_w_and_cut_time(self, time_cutoff='auto', visual=False):
        """ Function has same job as integrate_w but it also cuts off the
            unphysical data in the begininng and end of the simulation.
        """
        s_of_t = self.integrate_w(visual=visual)
        if time_cutoff=='auto':
            time_cutoff = self.find_time_cutoff(s_of_t)
        t_cut, s_of_t_cut = self.make_time_cut(s_of_t,
                                               time_cutoff=time_cutoff)
        return t_cut, s_of_t_cut


    def integrate_w_and_fft_t(self, visual=False):
        """ Function to return the fourier transform on the time variable t
            of S(w, t), after the integration on w (or omega) is completed.
        """
        # Get the data integrated on w (or omega)
        data_integrated = self.integrate_w(visual=visual)

        # Peform Fourier transform on this data
        times = np.loadtxt(self.timesfile)
        w, fw = self.fast_ft(times, data_integrated, visual=visual)

        return w, fw


    def find_time_cutoff(self, s_of_t):
        """ Function to automatically generate the time-cutoff for the data.
        """
        # Find the derivative of S_of_t
        der = np.gradient(s_of_t)
        # Add 1 to the derivative
        der += 1
        #fig = plt.figure(figsize=(3, 3))
        #gs = fig.add_gridspec(1, 1)
        #ax = fig.add_subplot(gs[0, 0])
        #ax.plot(der)
        #fig.savefig('data.pdf', bbox_inches='tight')
        # Compute when does this value not change by 10%
        for a in range(len(s_of_t)):
            criterion = np.abs(1 - der[a]) * 100
            if criterion < 0.001:
                break

        # Find the time when this happens
        times = np.loadtxt(self.timesfile)
        t_cut = times[a]

        return t_cut

    def find_ifft_cutoff(self):
        """ Function to automatically generate the inverse fourier cutoff.
        """
        wd_probe = self.wd_probe
        cutoff = self.cutoff
        ifft_cut = cutoff * np.sqrt(2) / wd_probe

        return ifft_cut


    def make_time_cut(self, s_of_t, time_cutoff=4):
        """ Function to return times after the cutoff is applied
            and s_of_t after the cut-off is applied.
        """
        # Find the position of the given time_cutoff
        times = np.loadtxt(self.timesfile)
        cutting_time = times[0] + time_cutoff
        pos = np.argmin(np.abs(times - cutting_time))

        # Find the cut-out time and the cut out data
        t_cut = times[pos:-pos]
        s_of_t_cut = s_of_t[pos:-pos]

        return t_cut, s_of_t_cut

    def one_by_probepulse_ft(self, w, ifft_cutoff=4):
        """ Function to elvaluate 1/ (fourier transform of probe pulse), this
            can be derived analytically and is given by exp(-w^2 sigma^2 / 4)
        """
        # We add a small value to avoid Gaussian becoming zero.
        den = np.exp( - ((w * self.wd_probe)**2) / 4) + 10**(-30)
        value = 1 / den

        # Kill all values above a cut-off
        kill_index = (np.abs(w) > ifft_cutoff).nonzero()
        value[kill_index] = 0

        return value


    def encode_equilibrium(self, t, s_of_t, eqb_time=10):
        """ Function to encode information of equilibrium before
            the action of pump pulse.
        """
        # Make the attach time
        dt = t[1] - t[0]
        ti = t[0] - eqb_time
        tf = t[0] - dt
        t_attach = np.arange(ti, tf+dt, dt)
        # Make sure the full t_new is always odd
        if np.mod(len(t_attach), 2) != 0:
            t_attach = np.arange(ti+dt, tf+dt, dt)

        # Make the attach s_of_t
        s_attach = np.ones(len(t_attach)) * s_of_t[0]
        # Attach to original data
        t_new = np.concatenate((t_attach, t))
        s_new = np.concatenate((s_attach, s_of_t))

        return t_new, s_new

    def encode_smooth(self,  t, s_of_t, eqb_time=100, beta=1):
        """ Function to encode a smooth fermi function prior to the
            the equilibrium data.
        """
        # Make the left attach time
        dt = t[1] - t[0]
        ti = t[0] - eqb_time
        tf = t[0] - dt
        t_l = np.arange(ti, tf+dt, dt)

        # Make the right attach time
        ti = t[-1] + dt
        tf = ti + eqb_time
        t_r = np.arange(ti, tf+dt, dt)

        # Make the left attach s_of_t
        s_of_t_l = np.ones(len(t_l)) * s_of_t[0]

        # Make the right attach of s_of_t
        s_of_t_r = np.ones(len(t_r)) * s_of_t[-1]

        # Attach the times together
        t_attach = np.concatenate((t_l, t, t_r))

        # Attach s_of_t together
        s_of_t_attach = np.concatenate((s_of_t_l,
                                        s_of_t,
                                        s_of_t_r))

        # Create the Smoothening function
        t0_l = t_attach[0] + eqb_time/2
        t0_r = t_attach[-1] - eqb_time/2
        fermi_l = 1 / ( np.exp(beta*(t_attach - t0_l)) + 1)
        fermi_r = 1 / ( np.exp(beta*(t_attach - t0_r)) + 1)
        smooth_fn = fermi_r - fermi_l

        # Make the new results
        t_new = t_attach
        s_new = s_of_t_attach * smooth_fn

        # Create the new times and s_of_t (making sure they are odd)
        M = len(t_attach)
        if np.mod(M, 2)==0:
            # If M is even, we make it odd by repeating the last
            # element one more time.
            s_odd = np.zeros(M+1)
            t_odd = np.zeros(M+1)
            s_odd[0:M] = s_new
            s_odd[M] = s_new[-1]
            t_odd[0:M] = t_new
            t_odd[M] = t_new[-1]
            t_new = t_odd
            s_new = s_odd

        return t_new, s_new


    def give_QFI(self, method='fft', eqb_time=100, beta=1,
                 visual=False, verbose=True, **kwargs):
        """ Function to return the quantum Fisher information (QFI) from
            the provided spectra.

        Paramaters:
        -----------
        method: string
            The method taken to calculate QFI.
        visual: bool
            Allows one to visualize steps taken to calculate the QFI
            in a given chosen method.
        verbose: bool
            Display the steps happening in the code.
        **kwargs: dict
            The variable number of arguments used in the method,
            passed as a dictionary.

        Returns:
        --------
        t_qfi: numpy 1D array
            The times at which QFI is calculated.
        qfi: numpy 1D array
            The corressponding QFI.

        """

        if method=='fft':
            """ This method accepts additional arguments as **kwargs

                Parameters:
                -----------
                time_cutoff: string or float
                    The number of units of time that must be ignored in the
                    begining and the end. If it is chosen as 'auto' then
                    the code automatically finds the time-cutoff, if it is
                    a number then it will be directly used as a cut-off.
                ifft_cutoff: string or float
                    Cutoff chosen for high-frequenies during inverse Fourier.
                    If it is chosen as 'auto' then the code automatically
                    choses the high-frequency cutoff. If it is a number, then
                    the number times (1/wd_probe) becomes the high-frequency
                    cutoff.
            """
            if verbose:
                print('----------------')
                print('Using FFT method')
                print('----------------')

            # Set default vaues of **kwargs or extract them
            if len(kwargs) == 0:
                kwargs = {'time_cutoff': 'auto', 'ifft_cutoff': 'auto'}
            time_cutoff = kwargs['time_cutoff']
            ifft_cutoff = kwargs['ifft_cutoff']

            # Integrate the spectra over frequency (w) to get time-dependent data
            if verbose: print('Integrating spectra over frequency ...')
            s_of_t = self.integrate_w(visual=visual)

            # Perform time cut-off process.
            if verbose: print('Cutting out unphysical times ...')
            if time_cutoff=='auto':
                time_cutoff = self.find_time_cutoff(s_of_t)
            t_cut, s_of_t_cut = self.make_time_cut(s_of_t,
                                     time_cutoff=time_cutoff)

            # Attach a longer equilibrium part
            if verbose: print('Encoding equilibrium information ...')
            t_cut, s_of_t_cut = self.encode_equilibrium(t_cut, s_of_t_cut,
                                     eqb_time=eqb_time)

            # Attach a smooth fermi-function (to remove Gibbs Phenomenon)
            if beta == None:
                if verbose: print('No soomthening is being performed ...')
            else:
                if verbose: print('Smoothing equilibrium using Fermi-function ...')
                t_cut, s_of_t_cut = self.encode_smooth(t_cut, s_of_t_cut,
                                                       eqb_time=eqb_time, beta=beta)

            # Peform Fourier transform on this data
            if verbose: print('Peforming Fourier transform (FT) ...')
            w, fw = self.fast_ft(t_cut, s_of_t_cut, visual=visual)

            # Obtain the inverted Gaussian prefactor
            if ifft_cutoff=='auto':
                ifft_cutoff = self.find_ifft_cutoff()
            prefactor = self.one_by_probepulse_ft(w,
                             ifft_cutoff=ifft_cutoff)

            # Get the product of the the two terms above
            Iw = prefactor * fw

            # Visualize Iw if asked for
            if visual:
                fig = plt.figure(figsize=(4, 3))
                gs = fig.add_gridspec(1, 1)
                ax = fig.add_subplot(gs[0, 0])
                ax.plot(w, np.real(Iw), color='k', label='Real')
                ax.plot(w, np.imag(Iw), color='C0', label='Imag')
                ax.legend(frameon=False)
                ax.set_xlabel(r"Frequency $\mathrm{\omega'}$")
                ax.set_ylabel(r"F($\omega'$)exp($\sigma_\mathrm{pr}^2\omega'^2/4$)")
                fig.savefig('./results/Fwgauss.pdf', bbox_inches='tight')

            # Do inverse FFT to get QFI
            if verbose: print('Performing deconvolution by inverse FT ...')
            t_qfi, qfi = self.fast_ift(t_cut, Iw)

            # Multiply by remanining prefactors
            if verbose: print('Finalizing and generating QFI ...')
            qfi *= (8 * np.sqrt(np.pi) * self.wd_probe)

            if self.dataorigin=='sqwt':
                qfi *= (1 / (8 * self.wd_probe * np.sqrt(np.pi)))
                if verbose: print('QFI successully generated.')
            elif self.dataorigin=='rixs':
                qfi *= 2*np.pi*(self.inverselife)**2 / (8 * self.wd_probe * np.sqrt(np.pi))
                if verbose: print('QFI successully generated.')
            else:
                if verbose: print('QFI successully generated.')

        elif method=='method2':
            print(' This method is not yet functional.')
        else:
            raise valueError('Unknown choice of integration for QFI')

        return t_qfi, np.real(qfi), w, Iw

