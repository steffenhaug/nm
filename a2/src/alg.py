import numpy as np
from numpy import linalg as LA
import sympy as sp

import scipy.sparse as scsp

MAX_ITER = 5000

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

    J  = symbolic_jac(F)

    # singular jacobian means trouble
    Jfn = callable_fn(J)
    assert LA.det(Jfn(x, y)) != 0

    # function-version of the Jacobian
    Ji = callable_fn(J.inv())

    def step(f, Ji_f, x):
        # computes the next iteration using the
        # Newton method equation.
        # r is the previous step
        return x - Ji_f(*x).dot(f(*x))

    for _ in range(MAX_ITER):
        px, py = x, y
        x,  y  = step(F, Ji, (x, y))
        yield x, y

        # check the tolerance criteria
        if LA.norm(F(x, y)) < tol:
            break
        if LA.norm((x - px, y - py)) < tol:
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

def solve_nd_fpi(M, N, f):
    # solves the linear system (M - N)x = b by
    # fix-point iteration x = inv(M)(Nx + b).

    # Compute inv(M), C and f
    Mi = LA.inv(M)
    C = Mi.dot(N)
    g = Mi.dot(f)

    assert spectral_radius(C) < 1

    u = g
    for _ in range(MAX_ITER):
        u = C.dot(u) + g
        yield u


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

def choose_matrices(A, method="jacobi", omega=1.0):
    # pick a method based on the parameter
    # i have included some redundant parameters
    # so it is possible to write "shorthand"
    # upper-case also works.
    M, N = {
        # Jacobi method
        "jacobi":        jacobi_mat,
        "j":             jacobi_mat,
        # Gauss-Seidel method
        "gauss-seidel":  gs_mat,
        "gs":            gs_mat,
        # Successive over-relaxation
        "sor": lambda A: sor_mat(A, omega),
    }[method.lower()](A)

    return M, N


def solve_nd(A, f, method="jacobi", omega=1.0, tol=1E-5):
    # solve Ax = b.
    # returns an iterator over tuples (u, r),
    # where u is successively better solutions,
    # and r is the residual vector.
    # omega is unused unless "sor" is specified

    M, N = choose_matrices(A,
                           method=method,
                           omega=omega)

    for i, u in enumerate(solve_nd_fpi(M, N, f)):
        # compute the residual
        r = f - A.dot(u)
        if LA.norm(r) < tol:
            break
    return i, u


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


def solve_nd_prec_fpi(C, g):
    # same as solve_nd, but requires precomputed matrices.
    u = g
    for _ in range(MAX_ITER):
        u = C.dot(u) + g
        yield u

def solve_nd_prec(A, f, method="jacobi", omega=1.0, tol=1E-5):
    M, N = choose_matrices(A,
                           method=method,
                           omega=omega)

    Mi = LA.inv(M)
    C = Mi.dot(N)
    g = Mi.dot(f)

    for i, u in enumerate(solve_nd_prec_fpi(C, g)):
        # compute the residual
        r = f - A.dot(u)
        if LA.norm(r) < tol:
            break
    return i, u
