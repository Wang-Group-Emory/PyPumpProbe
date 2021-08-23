import numpy as np
from babyrixs.operators import Ops

def observable(sys, tag='total_charge'):
    """ Function to return the matrix form of an observable

    Parameters:
    ----------
    sys: object of the pumpedchain class
        This contains the system to which your operator
        is associated.
    tag: string
        This is the tag which identifies what operator
        you neeed.

    Returns:
    --------
    opt_mtx: numpy 2D array
        This is the matrix form of the operator you are
        looking for.

    """
    N = sys.N
    dim = 4**N
    opt_mtx = np.zeros((dim, dim))
    cops = Ops(N)

    if tag=='total_charge':
        for i in range(N):
            ci_up = cops.get_mtx(site=i, sigma=0, type='an')
            ci_dn = cops.get_mtx(site=i, sigma=1, type='an')
            ci_dag_up = cops.get_mtx(site=i, sigma=0, type='cr')
            ci_dag_dn = cops.get_mtx(site=i, sigma=1, type='cr')
            ni = np.dot(ci_dag_up, ci_up) + np.dot(ci_dag_dn, ci_dn)
            opt_mtx += ni

    elif tag=='total_spin':
        for i in range(N):
            ci_up = cops.get_mtx(site=i, sigma=0, type='an')
            ci_dn = cops.get_mtx(site=i, sigma=1, type='an')
            ci_dag_up = cops.get_mtx(site=i, sigma=0, type='cr')
            ci_dag_dn = cops.get_mtx(site=i, sigma=1, type='cr')
            si = np.dot(ci_dag_up, ci_up) - np.dot(ci_dag_dn, ci_dn)
            opt_mtx += si

    elif tag=='charge_fluctuations':
        for i in range(N):
            ci_up = cops.get_mtx(site=i, sigma=0, type='an')
            ci_dn = cops.get_mtx(site=i, sigma=1, type='an')
            ci_dag_up = cops.get_mtx(site=i, sigma=0, type='cr')
            ci_dag_dn = cops.get_mtx(site=i, sigma=1, type='cr')
            ni = np.dot(ci_dag_up, ci_up) + np.dot(ci_dag_dn, ci_dn)
            opt_mtx += np.dot(ni, ni)

    elif tag=='spin_fluctuations':
        for i in range(N):
            ci_up = cops.get_mtx(site=i, sigma=0, type='an')
            ci_dn = cops.get_mtx(site=i, sigma=1, type='an')
            ci_dag_up = cops.get_mtx(site=i, sigma=0, type='cr')
            ci_dag_dn = cops.get_mtx(site=i, sigma=1, type='cr')
            si = np.dot(ci_dag_up, ci_up) - np.dot(ci_dag_dn, ci_dn)
            opt_mtx += np.dot(si, si)

    elif tag=='charge_i=1':
        i = 0
        ci_up = cops.get_mtx(site=i, sigma=0, type='an')
        ci_dn = cops.get_mtx(site=i, sigma=1, type='an')
        ci_dag_up = cops.get_mtx(site=i, sigma=0, type='cr')
        ci_dag_dn = cops.get_mtx(site=i, sigma=1, type='cr')
        opt_mtx = np.dot(ci_dag_up, ci_up) + np.dot(ci_dag_dn, ci_dn)

    elif tag=='charge_i=2':
        i = 1
        ci_up = cops.get_mtx(site=i, sigma=0, type='an')
        ci_dn = cops.get_mtx(site=i, sigma=1, type='an')
        ci_dag_up = cops.get_mtx(site=i, sigma=0, type='cr')
        ci_dag_dn = cops.get_mtx(site=i, sigma=1, type='cr')
        opt_mtx = np.dot(ci_dag_up, ci_up) + np.dot(ci_dag_dn, ci_dn)

    elif tag=='charge_fluctuation_at_site_i=1':
        i = 0
        ci_up = cops.get_mtx(site=i, sigma=0, type='an')
        ci_dn = cops.get_mtx(site=i, sigma=1, type='an')
        ci_dag_up = cops.get_mtx(site=i, sigma=0, type='cr')
        ci_dag_dn = cops.get_mtx(site=i, sigma=1, type='cr')
        ni = np.dot(ci_dag_up, ci_up) + np.dot(ci_dag_dn, ci_dn)
        opt_mtx = np.dot(ni, ni)

    else:
        raise ValueError('Unknow observable choice is entered.')

    return opt_mtx
