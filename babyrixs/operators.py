import numpy as np

class Ops:

    def __init__(self, N):
        """ A simple class to construct the creation
            and annhilation operators.

        Parameters
        ----------

        N : integer
            Total number of sites in the hubbard chain

        Attributes:
        -----------
        dim : integer
            The Hilbert space dimension of the operator.

        """

        self.N =  N
        self.dim = 4**N

        return


    def get_mtx(self, site=0, sigma=0, type='cr'):
        """ Function to return the matrix form of the field operators

        Parameters
        ----------
        site_no : integer
            It is the label of the site OR site number.
        sigma : integer
            Label for spin up = 0, spin down = 1
        type : string
            Label for creation or annihiliation
            operator. 'cr' = creation; 'an' = annhilitation

        Returns
        -------
        mtx : numpy 2D array
            matrix form of the operator you need.

        """
        N = self.N
        site_no = site
        c_up_dag = np.zeros((4, 4))
        c_dn_dag = np.zeros((4, 4))
        c_up_dag[1, 0] = 1
        c_up_dag[3, 2] = 1
        c_dn_dag[2, 0] = 1
        c_dn_dag[3, 1] = -1
        PM = np.diag(np.array([1, -1, -1, 1]))
        ID = np.diag(np.array([1,  1,  1, 1]))

        # For single site
        if N==1:
            if sigma == 0:
                if type == 'cr':
                    mtx = c_up_dag
                elif type == 'an':
                    mtx = c_up_dag.T
                else:
                    raise ValueError("Unknown type of operator chosen")
            else:
                if type == 'cr':
                    mtx = c_dn_dag
                elif type == 'an':
                    mtx = c_dn_dag.T
                else:
                    raise ValueError("Unknown type of operator chosen")

        # For two sites
        elif N==2:
            if sigma == 0:
                # site no 0
                if site_no == 0:
                    mtx = np.kron(c_up_dag, ID)
                    if type == 'an':
                        mtx = mtx.T
                # site no 1
                else:
                    mtx = np.kron(PM, c_up_dag)
                    if type == 'an':
                        mtx = mtx.T
            else:
                # site no 0
                if site_no == 0:
                    mtx = np.kron(c_dn_dag, ID)
                    if type == 'an':
                        mtx = mtx.T
                # site no 1
                else:
                    mtx = np.kron(PM, c_dn_dag)
                    if type == 'an':
                        mtx = mtx.T

        # For three sites
        elif N==3:
            if sigma == 0:
                # site no 0
                if site_no == 0:
                    mtx1 = np.kron(c_up_dag, ID)
                    mtx = np.kron(mtx1, ID)
                    if type == 'an':
                        mtx = mtx.T
                # site no 1
                elif site_no == 1:
                    mtx1 = np.kron(PM, c_up_dag)
                    mtx = np.kron(mtx1, ID)
                    if type == 'an':
                        mtx = mtx.T
                # site no 2
                else:
                    mtx1 = np.kron(PM, c_up_dag)
                    mtx = np.kron(PM, mtx1)
                    if type == 'an':
                        mtx = mtx.T
            else:
                # site no 0
                if site_no == 0:
                    mtx1 = np.kron(c_dn_dag, ID)
                    mtx = np.kron(mtx1, ID)
                    if type == 'an':
                        mtx = mtx.T
                # site no 1
                elif site_no == 1:
                    mtx1 = np.kron(PM, c_dn_dag)
                    mtx = np.kron(mtx1, ID)
                    if type == 'an':
                        mtx = mtx.T
                # site no 3
                else:
                    mtx1 = np.kron(PM, c_dn_dag)
                    mtx = np.kron(PM, mtx1)
                    if type == 'an':
                        mtx = mtx.T

        else:
            raise ValueError("The total number of sites cannot bigger that N = 3")

        return mtx


