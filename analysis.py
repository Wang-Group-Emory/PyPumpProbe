import numpy as np

from scipy import integrate
from scipy.fft import fft

class Analysis:

    def __init__(self, datafile, timesfile, omegafile):
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
        files_consistent: Bool
            If True, it means that the files are consistent in terms of the shape
            of the data.
        """
        self.datafile = datafile
        self.timesfile = timesfile
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


