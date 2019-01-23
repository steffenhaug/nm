import numpy as np
import sympy as sp

from numpy import linalg as la

MAX_ITER = 100

def symbolic_jac(py_fn):
    # computes a symbolic jacobian matrix.
    # assume that fnc: R^2 -> R^2.

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


def solve(F, x0, tol=10E-6):

    px, py = x0 # "previous x and y"

    assert px + py != 0, "singular Jacobian!"

    # compute the jacobian symbolically,
    # and create a callable version of
    # its inverse.
    J  = symbolic_jac(F)
    Ji = callable_fn(J.inv())

    def newton_step(f, Ji_f, r):
        # computes the next iteration using the
        # Newton method equation.
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
