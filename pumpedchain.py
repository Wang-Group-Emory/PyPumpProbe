import numpy as np
from constants import HBAR
from operators import Ops

from scipy.linalg import eigh

import matplotlib.pyplot as plt
import beauty.tanya


class Pumped1Dchain:
    """ Class keeping all the information of pumped 1D chain

        Parameters:
        -----------
    """
    def __init__(self, N=2,
                 gamma=1,
                 U=8,
                 Ne=1,
                 e0_pump=0,
                 om_pump=4.4,
                 wd_pump=3*HBAR,
                 t0_pump=20):
        """
        Initialization of class where default values are set to
        a Mott-antiferromagnetic chain from PRB 96, 235142 (2017).

        Parameters:
        -----------
        N: integer
            number of sites
        gamma: float
            hopping potential
        U: float
            Hubbard potential
        Ne: integer
            Filling or No. of electrons
        e0_pump: float
            Intensity of pump-pulse
        om_pump: float
            freqency of pump-pulse
        wd_pump: float
            freqncy of pump-pulse
        t0_pump: float
            Center of the pulse

        Attributes:
        -----------
        E0 : float
            Energy of the ground stateof the system.
        psiE0 : numpy 1D array
            Dirac vector of the ground state of the system.
        """

        self.N = N
        self.gamma = gamma
        self.U = U
        self.Ne = Ne
        self.e0_pump = e0_pump
        self.om_pump = om_pump
        self.wd_pump = wd_pump
        self.t0_pump = t0_pump
        # Set the state at t = 0 as the ground state
        # of the system.
        eg0, egs = self.get_gs(filling=Ne)
        self.E0 = eg0
        self.psiE0 = egs

        return

    def pump_pulse(self, t):
        """ Function for the pump pulse profile
        """
        e0_pump = self.e0_pump
        om_pump = self.om_pump
        wd_pump = self.wd_pump
        t0_pump = self.t0_pump

        time = (t - t0_pump)
        freq = om_pump/HBAR
        osc = np.cos(freq * time)
        gaussian = e0_pump * np.exp(-time**2 / (2*wd_pump**2))
        profile =  gaussian * osc

        return profile

    def get_ham(self, t=0, plot=False):
        """ Function to extract the Hamiltonian at a given time
        """
        N = self.N
        dim = 4**N
        gamma = self.gamma
        U = self.U

        ham = np.zeros((dim, dim), dtype='complex')
        # Get the operator objects
        c_ops = Ops(N)
        # Create lables for the sites and spins
        all_sites = np.arange(0, N)
        all_spins = np.array([0, 1])


        # Load the Nearest-Neighbor Hopping
        #-----------------------------------
        for a in range(N-1):
            for b, sigma in enumerate(all_spins):
                # Forward Nearest-Neigbor Hopping Potential
                profile = self.pump_pulse(t=t)
                f_hop_ij = - gamma * np.exp( 1j*profile )
                # Forward Nearest-Neighbour Hopping
                op_i = c_ops.get_mtx(site=a, sigma=sigma, type='cr')
                op_j = c_ops.get_mtx(site=a+1, sigma=sigma, type='an')
                cdag_i_cj = f_hop_ij * np.dot(op_i, op_j)
                # Add to Hamiltonian with the conjugate
                ham = ham +  cdag_i_cj +  np.conjugate(cdag_i_cj.T)

        # Load the Hubbard Term
        #----------------------
        for a in range(N):
            # Up spin number operator
            cdag_i_up = c_ops.get_mtx(site=a, sigma=0, type='cr')
            c_i_up = c_ops.get_mtx(site=a, sigma=0, type='an')
            n_i_up = np.dot(cdag_i_up, c_i_up)
            # Down spin number operator
            cdag_i_dn = c_ops.get_mtx(site=a, sigma=1, type='cr')
            c_i_dn = c_ops.get_mtx(site=a, sigma=1, type='an')
            n_i_dn = np.dot(cdag_i_dn, c_i_dn)
            # Added to Hamiltonian
            ham = ham + U*np.dot(n_i_up, n_i_dn)

        if plot:
            fig = plt.figure(figsize=(6, 2.7))
            gs = fig.add_gridspec(1, 2)
            ax = fig.add_subplot(gs[0, 0])
            bx = fig.add_subplot(gs[0, 1])
            ax.imshow(np.real(ham), aspect='auto', cmap='hot', vmin=0)
            bx.imshow(np.imag(ham), aspect='auto', cmap='hot', vmin=0)
            fig.savefig('ham.pdf', bbox_inches='tight')

        return ham

    def get_eigen(self, filling=1):

        """ Function to evaluate the eigen states of the unpumped
            Hamiltonian
        """
        N = self.N
        # Find eigenstates
        en, psi = eigh(self.get_ham(t=-50000))
        # Find the states where no. of electrons = filling
        dim = 4**N
        c_ops = Ops(N)
        n_op = np.zeros((dim, dim), dtype='complex')
        for a in range(N):
            o1 = c_ops.get_mtx(site=a, sigma=0, type='cr')
            o2 = c_ops.get_mtx(site=a, sigma=0, type='an')
            n_up = np.dot(o1, o2)
            o1 = c_ops.get_mtx(site=a, sigma=1, type='cr')
            o2 = c_ops.get_mtx(site=a, sigma=1, type='an')
            n_dn = np.dot(o1, o2)
            n_op = n_op + n_up + n_dn

        evals = []
        evecs = []
        for a in range(dim):
            vec = psi[:, a]
            vec_s = np.conjugate(vec.T)
            vec = np.dot(n_op, vec)
            ne = np.dot(vec_s, vec)
            ne = np.real(ne)
            if np.abs(ne - filling) < 10**(-6):
                evals.append(en[a])
                evecs.append(psi[:, a])

        evals = np.array(evals)
        evecs = np.array(evecs).T
        return evals, evecs

    def get_gs(self, filling=1):
        """ Function to get the groundstate
        """
        evals, evecs = self.get_eigen(filling=filling)
        sort_id = np.argsort(evals)
        evals = evals[sort_id]
        evecs = evecs[:, sort_id]
        e_gs = evals[0]
        gs = evecs[:, 0]

        return e_gs, gs

