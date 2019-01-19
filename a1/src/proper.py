# these actually return the correct answer,
# instead of repeatedly yielding better and
# better approximations.

def ddx(f, dx): return lambda x: (f(x + dx) - f(x)) / dx

MAX_ITER = 100

def bisect(f, I):
    a, b = I

    assert a < b
    assert f(a) * f(b) < 0

    for _ in range(MAX_ITER):
        m = 0.5 * (a + b)
        if f(a) * f(m) < 0:
            b = m
        elif f(b) * f(m) < 0:
            a = m
        else: break # m is 0, do nothing

    return m


def fpi(g, x0, tol=10E-6):
    # g contraction, finds x = g(x) by iteration
    xn    = g(x0) # x1
    xn_m1 = x0    # x0
    for _ in range(MAX_ITER):
        xn, xn_m1 = g(xn), xn

    return xn


def newton(f, dfdx, x0, tol=10E-6):
    g = lambda x: x - f(x) / dfdx(x)
    yield from fpi(g, x0)


# functions for problem 1
def I(h, r):
    p = (3 * r**2 * (r + h) - r**3 - h**3)
    q = (4 * r**3)
    return p / q

def g(h, r, rho):
    f1 = -4 * r
    f2 = (1 - 2*rho)/2
    f3 = 1 / (3 - (h/r)**2)
    return f1 * f2 * f3
