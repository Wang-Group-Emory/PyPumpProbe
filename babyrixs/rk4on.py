import numpy as np
from babyrixs.constants import HBAR
from babyrixs.mystates import custom_state

class Rk4on:

    def __init__(self, sys):
        """ Class to get the time-evolution operator based on RK4 scheme
        """
        self.sys = sys
        # Set initial time-evolution operator as U = 1
        N = sys.N
        dim = 4**N
        self.U = np.identity(dim, dtype='complex')

        # Now set the time-dependent Wave-function attribute
        psiE0 = sys.psiE0
        self.psi = np.dot(self.U, psiE0)

        return

    def U_rhs(self, t, U):
        """ RHS of the Schrodinger equation for time-evolution operator
        """
        sys = self.sys
        ham = sys.get_ham(t=t)
        rhs = (-1j/HBAR) * np.dot(ham, U)
        return rhs

    def propagate(self, t, dt):
        """ Function that updates the time-evolution operator
            as time evolves.
        """
        # Apply RK step 1
        U1 = self.U_rhs(t, self.U)*dt
        # Apply RK step 2
        U2 = self.U_rhs(t+dt/2, self.U + U1/2)*dt
        # Apply RK step 3
        U3 = self.U_rhs(t+dt/2, self.U + U2/2)*dt
        # Apply RK step 4
        U4 = self.U_rhs(t+dt, self.U + U3)*dt
        # Final psi
        self.U += (U1 + 2*U2 + 2*U3 + U4)/6

        return

    def heiesenberg(self, observable, tag='operator', save_in_file=False):
        """
        Function to evaluate a time-dependent operator associated
        to the system we are evolving.

        Parameters
        ----------
        observable: numpy 2D array
            This is the operator whoose Heiesenberg time-evolution
            is being calculated.
        tag: string
            This is a string that acts a tag for filename in which
            things are stored.
        save_in_file: Bool
            This option allows to save a file with the operator
            inside it.

        Returns:
        --------
        op_t : numpy 2D array
            The 2D numpy array of the operator.

        """
        U = self.U
        op_t = np.dot(observable, U)
        op_t = np.dot(np.conjugate(U.T), op_t)

        if save_in_file:
            filename = tag
            np.savetxt(filename, op_t.view(float))

        return op_t


    def expectation(self, observable, using='psi_t', my_state='my_state'):
        """
        Function to evaluate a time-dependent expectation value of
        an operator associated to the system we are evolving.

        Parameters
        ----------
        observable: numpy 2D array
            This is the operator whoose Heiesenberg time-evolution
            is being calculated.

        Returns:
        --------
        expectation_value : float
            Gives the expectation value of the operator provided.

        """
        if using=='psi_t':
            my_psi = self.psi
            my_psi_cj = np.conjugate(my_psi)
            psi1 = np.dot(observable, my_psi)
            expectation_value = np.dot(my_psi_cj, psi1)
            expectation_value =  np.real(expectation_value)
        elif using=='gs':
            my_psi = self.sys.psiE0
            my_psi_cj = np.conjugate(my_psi)
            psi1 = np.dot(observable, my_psi)
            expectation_value = np.dot(my_psi_cj, psi1)
            expectation_value =  np.real(expectation_value)
        elif using=='custom':
            my_psi = custom_state(my_state=my_state)
            my_psi_cj = np.conjugate(my_psi)
            psi1 = np.dot(observable, my_psi)
            expectation_value = np.dot(my_psi_cj, psi1)
            expectation_value =  np.real(expectation_value)
        else:
            raise ValueError('Unknown option supplied for calculating expectation value')

        return expectation_value
