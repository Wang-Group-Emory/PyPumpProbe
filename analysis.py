import numpy as np

from scipy import integrate
from scipy.fft import fft, ifft

import matplotlib.pyplot as plt

class Analysis:

    # TODO: Can we implement something to find wd_probe directly from the data.
    def __init__(self, datafile, timesfile, omegafile, wd_probe=0.5):
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
        self.wd_probe = wd_probe
        self.omegafile = omegafile
        self.files_consistent = self.check_data_consistency()

        return

    def check_data_consistency(self):
        data = np.loadtxt(self.datafile)
        times = np.loadtxt(self.timesfile)
        omega = np.loadtxt(self.omegafile)
        ny, nx = np.shape(data)

        if nx==len(times) and ny==len(omega):
            status = True
        else:
            raise valueError("The datafile for structure factor \
                              does not match the size of time or omega file.")
        return status

    #TODO: Check if 'ark' can be modified for better results
    def k_function(sel, Fk, N):
        """ Function to process Fk (discrete fourier transform) and return
            the continuous fourier transform fw.
        """
        k = np.arange(-N, N+1)
        ark = 1j * np.pi * k
        #ark = 1j * (2*np.pi) * (N/(2*N + 1)) * k * np.exp(1/(2*N) - 1/(4*N**2))
        prefactor = np.exp(ark)
        fw = prefactor * Fk
        return fw

    def fast_ft(self, ft):
        """ Function to perform Fourier transfrom of data in time-domain
            to the frequency-domain as well as the omega values.
        """
        # Check the length of the data and adjust it.
        M = len(ft)
        if np.mod(M, 2)==0:
            # If M is even, we make it odd by adding repeating the last
            # element one more time.
            ft_new = np.zeros(M+1)
            ft_new[0:M] = ft
            ft_new[M+1] = ft[-1]
            ft = ft_new
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
        times = np.loadtxt(self.timesfile)
        dt = times[1] - times[0]
        # Update fw with proper normalization factors
        fw = fw * dt
        # Present the corressponding omega values
        w = np.arange(-N, N+1) * ( np.pi / (N*dt) )

        return w, fw

    def fast_ift(self, Fk):
        """ Function to perform inverse Fourier transformation of data in
            frequency domain to the data in time-domain and also return the
            corressponding time values.
        """
        # Extract the value of M and N
        M = len(Fk)
        N = int((M - 1) / 2)
        # Use scipt ifft to get the inverse fourier transform
        fm = ifft(Fk)
        #---------------------------------
        # Processin part
        #---------------------------------
        left_values = np.flip(fm[1:N+1])
        centr_value = fm[0]
        rite_values = np.flip(fm[N+1:])
        fm[0:N] = left_values
        fm[N] = centr_value
        fm[N+1:] = rite_values
        #----------------------------------
        # Reverse the order of  fm
        #fm_rev = fm
        #--------------------------
        # Obtain the prefactor which goes as + 1, -1, + 1 ...
        prefactor = np.zeros(M)
        prefactor[::2] = 1
        prefactor[1::2] = -1
        fm  *= prefactor
        # Obtain the dt value
        times = np.loadtxt(self.timesfile)
        dt = times[1] - times[0]
        # Divide fm_rev by dt to get the final result
        ftn = fm/dt
        # Construct the time values
        tn = times
        return tn, ftn

    def integrate_w(self):
        """ Function to integrate the S(omega, t) data on omega
            and return the final sum as function of time. Note that
            it returns a 1D array which represents the values as a
            function of time once the integration on omega is complete.
        """
        # Find domega
        omega = np.loadtxt(self.omegafile)
        # Extract data and integrate
        data = np.loadtxt(self.datafile)
        data_integrated = integrate.simpson(data, omega, axis=0)
        return data_integrated

    def integrate_w_and_fft_t(self):
        """ Function to return the fourier transform on the time variable t
            of S(w, t), after the integration on w (or omega) is completed.
        """
        # Get the data integrated on w (or omega)
        data_integrated = self.integrate_w()
        # Peform Fourier transform on this data
        w, fw = self.fast_ft(data_integrated)

        return w, fw

    def one_by_probepulse_ft(self, w, eta=-6):
        """ Function to elvaluate 1/(fourier transform of probe pulse), this
            can be derived analytically and is given by exp(-w^2 sigma^2 / 4)
        """
        # We add a small value to avoid Gaussian becoming zero.
        value0 = np.exp( - ((w * self.wd_probe)**2) / 4)
        value0 = value0 + 10**(eta)
        value = 1 / value0
        return value

    def give_QFI(self, method='fft', visual=False, eta=-6):
        """ Function to return the quantum Fisher information from
            the provided spectra.
        """
        # Get result after integrating frequency w and doing FFT on time t
        w, fw = self.integrate_w_and_fft_t()

        # Obtain the inverted Gaussian prefactor
        prefactor = self.one_by_probepulse_ft(w, eta=eta)

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
            #ax.set_xlim([-12, 12])
            fig.savefig('./results/Fwgauss.pdf', bbox_inches='tight')

        # obtain QFI either by 'fft' method or 'sum' method
        if method=='fft':
            t_qfi, qfi = self.fast_ift(Iw)

        elif method=='sum':
            t_qfi = np.loadtxt(self.timesfile)
            exp_factor = np.exp( 1j*np.outer(w, t_qfi) )
            integrand = (exp_factor.T * Iw).T
            qfi = integrate.simpson(integrand, w, axis=0)
            qfi = qfi / (2*np.pi)

        else:
            raise valueError('Unknown choice of integration for QFI')

        # Muliply by the remaining prefactors
        qfi *= (8 * np.sqrt(np.pi) * self.wd_probe)

        return t_qfi, qfi



