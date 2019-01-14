import src.alg as alg
import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib

from math import sqrt
import random as rand

font = {'family' : 'normal',
        'size'   : int(12 / 0.7) + 1}

matplotlib.rc('font', **font)

# 2.a d cos dx
f = plt.figure(constrained_layout=True)
def gen_data(deltas):
    for dx in deltas:
        dcos_dx = alg.ddx(cos, dx)
        approx = dcos_dx(pi/4)
        exact  = -sin(pi/4)
        yield (
            abs(approx - exact),
            dx
        )

dxs = (1/2**k for k in range(1, 25))
data = list(gen_data(dxs))
error, delta = zip(*data)

plt.yscale("log", basey=2)
plt.xscale("log", basex=2)

plt.plot(delta, error, "k")
plt.gca().invert_xaxis()


plt.xlabel("$\Delta x")
plt.ylabel("$\mathtt{error(}\Delta x\mathtt{)}$")

f.savefig("approx_ddx.pgf")

# 2.b sqrt(sqrt(...sqrt(x)))
f = plt.figure(constrained_layout=True)

def iter_sqrt(N):
    x  = rand.random() * 100
    xh = x

    for l in range(1, N):
        for k in range(1, l):
            xh = sqrt(xh)
        for k in range(1, l):
            xh = xh**2

        e  = abs(x - xh)
        er = abs(x - xh) / abs(x)

        yield (e, er, l)

data = list(iter_sqrt(N = 100))
es, ers, ls = zip(*data)

plt.yscale("log")

plt.xlabel("$l$")
plt.ylabel("$\mathtt{error(}l\mathtt{)}$")

plt.plot(ls, es, color="r", label="$e(l)$")
plt.plot(ls, ers, color="g", label="$er(l)$")

plt.legend()

f.savefig("sqrt_err.pgf")


def g(x): return x**5 - 4*x + 2
def dgdx(x): return 5*x**4 - 4

n = list(alg.newton(g, dgdx, 1))
b = list(alg.bisect(g, (1, 5)))

# 3. speed of convergence abs tol
f = plt.figure(constrained_layout=True)
plt.yscale("log")

def abs_err(data):
    for xn, xn_p1 in alg.pairs(iter(data)):
        yield abs(xn - xn_p1)

i, err_abs_n = zip(*list(enumerate(abs_err(n))))
_, err_abs_b = zip(*list(enumerate(abs_err(b))))

plt.plot(i, err_abs_b, color="red", label="bisect")
plt.plot(i, err_abs_n, color="green", label="newton")

plt.legend()

f.savefig("speed_conv_abs.pgf")

# 3. speed of conv. f tol
f = plt.figure(constrained_layout=True)
plt.yscale("log")


def f_err(f, data):
    for xn in data:
        yield abs(f(xn))

j, err_f_n = zip(*list(enumerate(f_err(g, n))))
_, err_f_b = zip(*list(enumerate(f_err(g, b))))

plt.plot(j, err_f_b, color="red", label="bisect")
plt.plot(j, err_f_n, color="green", label="newton")


plt.legend()

f.savefig("speed_conv_f.pgf")

