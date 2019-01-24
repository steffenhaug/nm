import numpy as np
from numpy import linalg as LA
import sympy as sp

from numpy import linalg as la
import scipy.sparse as scsp

MAX_ITER = 100

def symbolic_jac(py_fn):
    # computes a symbolic jacobian matrix.
    # assume that py_fn : R^2 -> R^2.

    x, y = sp.symbols('x y')

    # compute the entries of the vector by
    # evaluating the function for sp-symbols
    f1, f2 = py_fn(x, y)
    F  = sp.Matrix([f1, f2])

    # compute the jacobian
    J = F.jacobian([x, y])

    return J


def callable_fn(symbolic):
    x, y = sp.symbols('x y')
    return sp.lambdify([x, y], symbolic, 'numpy')


def solve(F, x0, tol=1E-6):

    px, py = x0 # "previous x and y"

    assert px + py != 0, "Invalid starting point; singular Jacobian!"

    # compute the jacobian symbolically,
    # and create a callable version of
    # its inverse.
    J  = symbolic_jac(F)
    Ji = callable_fn(J.inv())

    def newton_step(f, Ji_f, r):
        # computes the next iteration using the
        # Newton method equation.
        # f : R2 -> R2,
        # Ji_f is the invers Jacobian of f
        # r is the previous step
        return r - Ji_f(*r).dot(f(*r))

    x, y = newton_step(F, Ji, x0)

    for _ in range(MAX_ITER):
        # check the tolerance criteria
        if la.norm(F(x, y)) < tol:
            break
        if la.norm((x - px, y - py)) < tol:
            break

        px, py = x, y
        x, y  = newton_step(F, Ji, (x, y))

    return x, y

def A(k, n):
    B = scsp.diags([1, -4, 1], [-1, 0, 1], shape=(n, n))
    I = scsp.eye(n)
    upperones = scsp.diags([1, 1], [-n, n], shape=(n**2, n**2))
    L = scsp.kron(I, B) + upperones

    dx = 1/n
    
    A = L + dx**2*k**2*scsp.eye(n**2)

    return A.toarray()

def rho(M):
    # compute the spectral radius of a matrix
    #   rho(M) = max |l_i|
    #         0 <= i < n
    # where l_i is eigenvalue number i.
    return np.max(np.abs(LA.eigvals(M)))

def solve_nd_fpi(M, N, b, tol=1E-6):
    # assumes M, N are numpy arrays.
    # solves the linear system Ax = b by iteration:
    #   Ax = b
    #   (M - N)x = b
    #   Mx = Nx + b
    #   x = inv(M)(Nx + b)
    # is a contraction if C = inv(M)N has spectral radius
    # rho(C) < 1 by the banach fixed point theorem. obviously
    # we need to have A = M - N.

    # Compute C and f
    Mi = np.invert(M)
    C = Mi * N
    f = Mi * b

    assert rho(C) < 1, "Iteration diverges!"

    def g(C, f, x):
        # compute one step of the iteration
        return C*x + f

    
    x = g(C, f, b)


