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

    x, y = x0
    assert x + y != 0, "Invalid starting point; singular Jacobian!"

    # compute the jacobian symbolically,
    # and create a callable version of
    # its inverse.
    J  = symbolic_jac(F)
    Ji = callable_fn(J.inv())

    def step(f, Ji_f, r):
        # computes the next iteration using the
        # Newton method equation.
        # f : R2 -> R2,
        # Ji_f is the invers Jacobian of f
        # r is the previous step
        return r - Ji_f(*r).dot(f(*r))


    for _ in range(MAX_ITER):
        px, py = x, y
        x,  y  = step(F, Ji, (x, y))
        yield x, y

        # check the tolerance criteria
        if la.norm(F(x, y)) < tol:
            break
        if la.norm((x - px, y - py)) < tol:
            break

def last(it):
    # run an iterator to the end
    x = None
    for x in it: pass
    return x


def A(k, n):
    B = scsp.diags([1, -4, 1], [-1, 0, 1], shape=(n, n))
    I = scsp.eye(n)
    upperones = scsp.diags([1, 1], [-n, n], shape=(n**2, n**2))
    L = scsp.kron(I, B) + upperones

    dx = 1/n

    A = L + dx**2 * k**2 * scsp.eye(n**2)

    return A.toarray()

def spectral_radius(M):
    return np.max(np.abs(LA.eigvals(M)))

def solve_nd_fpi(M, N, f, tol=1E-6):
    # solves the linear system (M - N)x = b by
    # fix-point iteration x = inv(M)(Nx + b).

    # Compute inv(M), C and f
    Mi = LA.inv(M)
    C = Mi.dot(N)
    g = Mi.dot(f)

    assert spectral_radius(C) < 1, "Spectral radius is too big!"

    x = g
    for _ in range(MAX_ITER):
        x = C.dot(x) + g

    return x

def solve_nd(A, b, tol=1E-6, method="jacobi", omega=1):
    # solve Ax = b
    # omega is only relevant if you choose
    # the method successive over-relaxation

    def jacobi_mat(A):
        # Jacobi method
        M = np.diag(A.diagonal())
        N = M - A
        return M, N

    def gs_mat(A):
        # Gauss-Seidel
        M = np.tril(A)
        N = M - A
        return M, N

    def sor_mat(A, omega):
        # successive over-relaxation
        D = np.diag(A.diagonal())
        L = np.tril(A, k=-1)
        M = D + omega*L
        N = M - A
        return M, N

    # pick a method based on parameter
    # i have included some redundant parameters
    # so it is possible to write "shorthand"
    M, N = {
        "jacobi":        jacobi_mat,
        "j":             jacobi_mat,
        "gauss-seidel":  gs_mat,
        "gs":            gs_mat,
        "sor": lambda A: sor_mat(A, omega),
    }[method.lower()](A)

    x = solve_nd_fpi(M, N, b, tol=tol)

    return x

def lattice(n):
    # produces a sequence of n^2 points
    # in an n x n lattice over the domain
    #   omega = [0, 1] x [0, 1],
    # including the .
    for j in range(n):
        for i in range(n):
            yield i/(n - 1), j/(n - 1)

def sample(F, n):
    # samples F over a n x n lattice.
    return np.array([F(x,y) for x, y in lattice(n)])
