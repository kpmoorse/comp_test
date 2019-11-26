import numpy as np
import itertools
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sindy(Xdot, Thx, thresh):

    Xi = np.empty((Thx.shape[1],0))

    for xdot in Xdot.T: # loop over regressand columns

        # U, S, V = np.linalg.svd(Thx, full_matrices=False)
        # xi = V.dot(np.linalg.inv(np.diag(S))).dot(U.T).dot(xdot)
        xi = np.linalg.lstsq(Thx, xdot)[0]
        xi[np.abs(xi)<1e-5] = 0
        Xi = np.hstack((Xi, xi[:,None]))

    return Xi


# Generate nonlinear polynomial combinations of columns
def nlp_col(A, order=5):

    n, m = A.shape
    B = np.empty((n,0))
    sym = []

    for i in range(order+1):

        for combination in itertools.combinations_with_replacement(A.T, i):

            new = np.ones((n,1))

            for factor in combination:

                new = np.multiply(new, factor[:,None])

            B = np.hstack((B,new))

    return B


if __name__ == '__main__':

    # Lorenz differential equation
    def differential(state, t, coeffs):
        p,s,b = coeffs
        x,y,z = state
        return s*(y-x), x*(p-z)-y, x*y-b*z

    # Simulate Lorenz eqn and return state and derivatives
    def sim_lorenz(tt=np.arange(0,100,0.01), coeffs=(28,10,8./3)):

        def f(state, t):
            return differential(state, t, coeffs)

        state0 = [1., 1., 1.]
        states = odeint(f, state0, tt)
        dstates = np.array([differential(s, None, coeffs) for s in states])

        return states, dstates

    # Simulate lorenz and generate polynomila library
    coeffs = (28,10,8./3)
    states, dstates = sim_lorenz(coeffs=coeffs) 
    library = nlp_col(states, order=5)

    # Run SINDy and print results
    Xi = sindy(dstates, library, 0)
    print(Xi)
    