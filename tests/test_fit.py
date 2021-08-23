import numpy as np
from babyrixs.fit import Fit

# Construct a function to fit
t = np.linspace(-5, 5, 1000)
R_of_t = np.sin(8*t) * np.exp(-t**2)

# Use fit class to perform the fit on data and return A_i, P_i, and W_i
make_fit = Fit(R_of_t, t)
A_i, P_i, W_i = make_fit.give_Ai(N=10, visual=True)


