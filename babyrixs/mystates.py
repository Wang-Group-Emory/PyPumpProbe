import numpy as np

def custom_state(my_state='one_particle_at_i=1'):
    if my_state=='one_particle_at_i=1':
        psi1 = np.array([0, 1, 0, 0])
        psi2 = np.array([1, 0, 0, 0])
        state = np.kron(psi1, psi2)
    elif my_state=='one_particle_at_i=2':
        psi1 = np.array([1, 0, 0, 0])
        psi2 = np.array([0, 1, 0, 0])
        state = np.kron(psi1, psi2)
    else:
        raise ValueError('Unknown option of custom state chosen.')

    return state
