from math import cos, sin

def ddx(f, dx): return lambda x: (f(x + dx) - f(x)) / dx



MAX_ITER = 100

def bisect(f, I):
    a, b = I

    assert a < b
    assert f(a) * f(b) < 0

    for _ in range(MAX_ITER):
        m = 0.5 * (a + b)
        yield m
        if f(a) * f(m) < 0:
            b = m
        elif f(b) * f(m) < 0:
            a = m
        else: return # f(m) == 0 so stop


def fpi(g, x0, tol=10E-6):
    # g contraction, finds x = g(x) by iteration
    # starting with x1
    xn    = g(x0) # x1
    xn_m1 = x0    # x0
    for _ in range(MAX_ITER):
        yield xn
        xn, xn_m1 = g(xn), xn


def newton(f, dfdx, x0, tol=10E-6):
    g = lambda x: x - f(x) / dfdx(x)
    yield from fpi(g, x0)

def pairs(it):
    # this method is deprecated, but w/e
    x = next(it)
    y = next(it)
    while True:
        yield(x, y)
        x, y = y, next(it)
