from math import cos, sin


MAX_ITER = 100

def ddx(f, dx):
    return lambda x: (f(x + dx) - f(x)) / dx


def bisect(f, I, tol=10E-6):
    a, b = I

    assert a < b
    assert f(a) * f(b) < 0

    for _ in range(MAX_ITER):
        if abs(a - b) < tol: break
        m = 0.5 * (a + b)
        if   f(a) * f(m) < 0:
            b = m
        elif f(b) * f(m) < 0:
            a = m
        else: # then f(m) must be zero
            return m

    return m


def fpi(g, x0, tol=10E-6):
    # g contraction, finds x = g(x) by iteration
    xn    = g(x0)
    xn_m1 = x0
    for _ in range(MAX_ITER):
        if abs(xn - xn_m1) < tol: break
        xn, xn_m1 = g(xn), xn

    return xn


def newton(f, dfdx, x0, tol=10E-6):
    g = lambda x: x - f(x) / dfdx(x)
    return fpi(g, x0, tol)
