import numpy as np
from scipy.optimize import least_squares
from scipy import integrate

import matplotlib.pyplot as plt
import beauty.tanya

class Fit:

    def __init__(self, R_of_t, t):
        """ Class to perform fitting of a given function R_of_t

        Parameters:
        -----------
        R_of_t: numpy 1D array
            The dataset to be fitted.
        t: numpy 1D array
            The corressponding x-values or times

        Attributes:
        -----------
        Same as above
        """
        self.R_of_t = R_of_t
        self.t = t

        return

    def gaussian(self, time, P, W):
        """ The Gaussian function we are using to fit. Note they
            are not normalized.
        """
        gauss = np.exp(- (time - P)**2 / (2*W**2) )
        return gauss

    def sum_of_gaussians(self, time, R_i, P_i, W_i):
        """ Function to return the sum of Gaussians with
            coefficients R_i, positions P_i, and width
            W_i.
        """
        N = len(R_i)
        sum_of_gaussians = np.zeros(len(time))
        for a in range(N):
            single_gaussian = R_i[a] * self.gaussian(time, P_i[a], W_i[a])
            sum_of_gaussians += single_gaussian

        return sum_of_gaussians

    def cost_function(self, x):
        """ The cost function. This takes the data to be fitted,
            subtracts the sum of Gaussians, and the returns the
            sum of the square of the resulting difference.

        Parameters:
        -----------
        x: numpy 1D array
            Contains R_i, P_i, and W_i put together as a numpy
            1D array of size 3N.

        Returns:
        --------
        cost_fn: float
            The cost function described above.

        """
        # Slice out the parameters
        N = int(len(x) / 3)
        R_i = x[0:N]
        P_i = x[N:2*N]
        W_i = x[2*N:3*N]

        # Create the cost function.
        t = self.t
        R_of_t = self.R_of_t
        sum_of_gaussians = self.sum_of_gaussians(t, R_i, P_i, W_i)
        cost_fn = np.sum( (R_of_t - sum_of_gaussians)**2 )

        return cost_fn

    def initialize_parameters(self, N, initiate_style='uniform'):
        """ Function to initalize the fit parameters
        """
        if initiate_style=='uniform':
            # Find maxima of the given data
            max_R_of_t = np.max(self.R_of_t)
            # Find the initial and final point
            ti = self.t[0]
            tf = self.t[-1]

        elif initiate_style=='smart':
            t = self.t
            R_of_t = self.R_of_t
            # Find the largest deviation from zero in data
            max_R_of_t = np.max(np.abs(R_of_t))
            # Start from left corner and find when does
            # deviation become 10% of the largest.
            for a, time in enumerate(t):
                dev = np.abs(R_of_t[a]) / max_R_of_t
                criterion = 100 * dev
                if criterion > 20:
                    break
            ti = time

            for a, time in enumerate(t):
                dev = np.abs(R_of_t[-a]) / max_R_of_t
                criterion = 100 * dev
                if criterion > 20:
                    break
            tf = t[-a]
            print(ti)
            print(tf)
        else:
            raise valueError('Unknow option for initiate style chosen.')

        # Equaly distaned Gaussians
        R_i = np.zeros(N); R_i[0::2] = 1; R_i[1::2] = -1; R_i *= max_R_of_t
        P_i = np.linspace(ti, tf, N)
        W_i = 0.5*np.ones(N)

        return R_i, P_i, W_i

    def SumOfGaussians(self, N=15, initiate_style='smart',
                       visual=False, verbose=True):
        """ Function to return the coefficients R_i, position P_i,
            and the width W_i of the Gaussians fitted on the
            supplied data.

        Parameters:
        -----------
        N: The number of Gaussian functions being used.

        Returns:
        --------
        R_i: float
            The coefficients of the Gaussian functions.
        P_i: float
            The position at which the Gaussian is centered.
        W_i: float
            The width of the Gaussian functions.
        initiate_style: string
            Choose how you want to initiate your parameters.
        visual: bool
            Option to view the fit performed
        verbose: bool
            Print to see what is happening in the code.

        Note:
        -----
        'i' represents the index of the Gaussian functions, therefore,
        i = 1, 2, 3, ..., N.

        """
        #----------------------------------------------------------------
        # Set the initial values for the parameters. We will assume
        # equally spaced Gaussians of identical width. R_i coefficents
        # will change sign alternatively +, -, + , - , ...
        #----------------------------------------------------------------

        # Find maxima of the given data
        max_R_of_t = np.max(self.R_of_t)
        # Find the initial and final point
        ti = self.t[0]
        tf = self.t[-1]

        # Initiate fitting parameters
        R_i, P_i, W_i = self.initialize_parameters(N,
                        initiate_style=initiate_style)
        X_i = np.zeros(3*N)
        X_i[0:N] = R_i
        X_i[N:2*N] = P_i
        X_i[2*N:3*N] = W_i

        # Optimize the parameters using scipy
        if verbose:
            print('--------------------------')
            print(' Optimizing fit parameters')
            print('--------------------------')
        residuals = least_squares(self.cost_function, X_i)
        X_i = residuals.x
        R_i = X_i[0:N]
        P_i = X_i[N:2*N]
        W_i = X_i[2*N:3*N]

        # Visualize the fit performed if needed
        if visual:
            t = self.t
            R_of_t = self.R_of_t
            our_fit = self.sum_of_gaussians(t, R_i, P_i, W_i)
            fig = plt.figure(figsize=(5, 5))
            gs = fig.add_gridspec(2, 1, hspace=0.3)
            ax = fig.add_subplot(gs[0, 0])
            bx = fig.add_subplot(gs[1, 0])
            ax.plot(t, R_of_t, color='C1', label='Function', lw=2)
            ax.plot(t, our_fit, color='k', ls=':', label='Fit', lw=2)
            ax.legend(frameon=False)
            ax.set_ylabel('Function and its Fit')
            ax.set_xlim([t[0], t[-1]])

            for a in range(N):
                bx.plot(t, R_i[a]*self.gaussian(t, P_i[a], W_i[a]), lw=2)
            bx.set_xlim([t[0], t[-1]])
            bx.set_xlabel(' Time t')
            bx.set_ylabel(' Gaussians ')
            fig.savefig('./results/gaussian_fit.pdf', bbox_inches='tight')

        return R_i, P_i, W_i

    # TODO: Test this
    def overlap(self, P1, P2, W1, W2):
        """ Function to return the overlap-integral of two Gaussians.
        """
        # Find out which is greater
        P = np.array([P1, P2])
        W = np.array([W1, W2])
        pos_max = np.argmax(P)
        pos_min = np.argmin(P)
        max_p = P[pos_max]; min_p = P[pos_min]
        max_w = W[pos_max]; min_w = W[pos_min]

        # Find the integration region
        ti = min_p - 10*max_w
        tf = max_p + 10*max_w
        dt = min_w/30
        time = np.arange(ti, tf+dt, dt)

        # Integrate the Gaussians product numerically
        g1 = self.gaussian(time, P1, W1)
        g2 = self.gaussian(time, P2, W2)
        g1g2_integrate = integrate.simpson(g1*g2, time)

        return g1g2_integrate

    # TODO: Test this
    def convoluted_overlap(self, P1, P2, W1, W2, sigma=1):
        """ Function to evaluate the convoluted overlap
            of two gaussian functions.
        """
        # Make the function
        def three_gaussians(t, tau):
            g1 = self.gaussian(t, P1, W1)
            g2 = self.gaussian(t, tau, sigma)
            g3 = self.gaussian(tau, P2, W2)
            g1g2g3 = (1 / (sigma*np.sqrt(2*np.pi)) ) * (g1*g2*g3)
            return g1g2g3
        # Set the integration limits
        max_w = np.max(np.array([W1, W2, sigma]))
        tau_i = P2 - 10*max_w; tau_f = P2 + 10*max_w
        t_i = P1 - 10*max_w; t_f = P1 + 10*max_w
        res = integrate.nquad(three_gaussians, [[t_i, t_f], [tau_i, tau_f]])
        Gab = res[0]

        return Gab

    # TODO: Test this
    def give_SandGmatrix(self, N=10, sigma=1, initiate_style='smart',
                         visual=False, verbose=False):
        """ Function to return the overlap S-matrix
        """
        # Obtain the Fit paramaters
        R_i, P_i, W_i = self.SumOfGaussians(N=N,
                        initiate_style=initiate_style,
                        visual=visual,
                        verbose=verbose)

        # Cereate the matrix
        S = np.zeros((N, N))
        G = np.zeros((N, N))
        for a in range(N):
            for b in range(N):
                P1 = P_i[a]; W1 = W_i[a]
                P2 = P_i[b]; W2 = W_i[b]
                S[a, b] = self.overlap(P1, P2, W1, W2)
                G[a, b] = self.convoluted_overlap(P1, P2, W1, W2, sigma=sigma)

        return S, G, R_i, P_i, W_i

    # TODO: Write better comments and test this
    def give_Ai(self, N=10, sigma=1, initiate_style='smart',
                visual=False, verbose=False):
        """ Function to return coefficents A_i, P_i, and W_i
        """
        # Obtain S, G, matrix and P_i and W_i
        S, G, R_i, P_i, W_i = self.give_SandGmatrix(N=N, sigma=sigma,
                         initiate_style=initiate_style,
                         visual=visual,
                         verbose=verbose)
        # Find the Ai coefficients
        A_i = np.dot(S, R_i)
        A_i = np.dot( np.linalg.inv(G), A_i )

        return A_i, P_i, W_i
