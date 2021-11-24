import numpy as np

class RixsAnalysis:

    def __init__(self, rixsfile,
                       rixstime,
                       rixsomega,
                       rixsomg_i):
        """ Class to manipulate the time-resolved RIXS data. The purpose here is
            to collect all functions needed to process the RIXS data.

        Parameters:
        -----------
        rixsfile: string
            The name (or location) of the .txt file which contains the time-resolved RIXS
            data.
        rixstime: string
            The anme of the .txt file which contains the times used in the spectra
            generation.
        rixsomega: string
            The name of the .txt file which contains the omega values used in the
            spectra generation.
        rixsomg_i:
            The name of the file that contains the w_in (incident X-ray photon energy)
            values used in the spectra generation.

        """

        self.rixsfile = rixsfile
        self.rixstime = rixstime
        self.rixsomega = rixsomega
        self.rixsomg_i = rixsomg_i

        return

    def give_rixs_SF(self,
                     w_in_choose=0,
                     savefile=True,
                     filename='rixs_SF.txt'):
        """ Funtion to ingest the file with rixs data and the spit out the
        structure factor like data file for the give w_in (incoming frequency)

        Parameters:
        -----------
        rixsfile: string
            The name of the .txt file that contains the RIXS data. For the users
            of ED code (built by Yao Wang), this is the data file you get from the
            code (typically named as trRIXS.txt)
        rixstime: string
            The name of the .txt file which contains the times used in the
            trRIXS spectra generation.
        rixsomega: string
            The name of the .txt file which contains the omega values used in
            the spectra generation.
        rixsomg_i: string
            The name of the .txt file which contains the list of w_in (incoming
            frequency) that are used in the generation of trRIXS spectra.
        w_in_choose: float
            This is the choice of the frequency which corressponds to the resonance.
            The resonance is where the trRIXS spectra closely resembles the
            structure factor.
        savefile: Bool
            Option to choose to save the output into a file.
        filename: string
            Name of the file into which the extracted data will be stored.

        Returns:
        --------
        rixs_SF: numpy array
            The 2D numpy array which picks the structure factor like data
            from the the trRIXS data
        """

        rixsfile = self.rixsfile
        rixsomega = self.rixsomega
        rixsomg_i = self.rixsomg_i

        print('Loading trRIXS data ...')
        rixsdata = np.loadtxt(rixsfile)

        # Load the omega_in and omega data
        omg_in = np.loadtxt(rixsomg_i)
        omg_l = np.loadtxt(rixsomega)

        # Data as a function of time for w_in = 1.8
        pos_in = np.argmin(np.abs(omg_in - w_in_choose))
        pos_i = pos_in*len(omg_l)
        pos_f = pos_i + len(omg_l)
        rixs_SF = rixsdata[:, pos_i:pos_f]

        # Save the file for usage
        if savefile:
            np.savetxt(filename, rixs_SF)

        return rixs_SF

