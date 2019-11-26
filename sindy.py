import numpy as np
import itertools
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, S

# Mulit-input, multi-output sparse linear regression via iterated thresholding
def sindy(Xdot, Thx, labels, thresh):

    # Xi = [np.empty((Thx.shape[1],0))]
    Xi = []
    lb = []

    for xdot in Xdot.T: # loop over regressand columns

        regset = np.ones(Thx.shape[1]).astype(bool)
        flag = True

        while flag:

            xi = np.linalg.lstsq(Thx[:,regset], xdot, rcond=None)[0]

            temp = regset
            regset = np.zeros(Thx.shape[1]).astype(bool)
            regset[temp] = np.abs(xi)>thresh
            if np.all(temp == regset):
                flag = False
            # flag = False
            # Xi = np.hstack((Xi, xi[:,None]))

        Xi.append(xi)
        lb.append([labels[i] for i in np.argwhere(regset).flatten()])

    return Xi, lb


# Generate nonlinear polynomial combinations of columns
def nlp_col(A, order=5):

    n, m = A.shape
    B = np.empty((n,0))
    sym = []

    sym_list = [symbols('x{}'.format(i)) for i in range(m)]

    for i in range(order+1):

        for combination in itertools.combinations_with_replacement(range(m), i):

            new_sym = S(1)
            new_val = np.ones(n)

            for factor in combination:

                new_sym = new_sym * sym_list[factor]
                new_val = new_val * A[:,factor]

            B = np.hstack((B,new_val[:,None]))
            sym.append(str(new_sym))

    return B, sym


if __name__ == '__main__':

    # A = np.array([[1,2],[3,4],[5,6]])
    # print(nlp_col(A, order=2))

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

    # Simulate lorenz and generate polynomial library
    coeffs = (28,10,8./3)
    states, dstates = sim_lorenz(coeffs=coeffs)
    states += np.random.normal(0, 0.5, states.shape)
    dstates += np.random.normal(0, 5, dstates.shape)
    library, labels = nlp_col(states, order=5)

    # Run SINDy and print results
    Xi, lb = sindy(dstates, library, labels, 0.25)
    print(Xi)
    print(lb)

    fig = plt.figure()
    plt.subplot(121, projection='3d')
    plt.plot(states[:,0], states[:,1], states[:,2])
    plt.subplot(122, projection='3d')
    plt.plot(dstates[:,0], dstates[:,1], dstates[:,2])
    plt.show()
