import src.alg as alg
import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib

# plot style and font setup
plt.style.use('classic')
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

plt.plot(delta, error, color="red")
plt.gca().invert_xaxis()


plt.xlabel("$\Delta x")
plt.ylabel("$\mathtt{error(}\Delta x\mathtt{)}$")

f.savefig("approx_ddx.pgf")

# 2.b sqrt(sqrt(...sqrt(x)))
f = plt.figure()

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

plt.plot(ls, es, color="red", label="$e(l)$")
plt.plot(ls, ers, color="blue", label="$er(l)$")

plt.legend(loc="upper left")

f.savefig("sqrt_err.pgf")


def g(x): return x**5 - 4*x + 2
def dgdx(x): return 5*x**4 - 4

n = list(alg.newton(g, dgdx, 1))
b = list(alg.bisect(g, (1, 5)))

# 3. speed of convergence abs tol
f = plt.figure()
plt.yscale("log")

def abs_err(data):
    for xn, xn_p1 in alg.pairs(iter(data)):
        yield abs(xn - xn_p1)

i, err_abs_n = zip(*list(enumerate(abs_err(n))))
j, err_abs_b = zip(*list(enumerate(abs_err(b))))

plt.plot(i, err_abs_n, color="blue", label="newton")
plt.plot(j, err_abs_b, color="red", label="bisect")

plt.legend()

f.savefig("speed_conv_abs.pgf")

# 3. speed of conv. f tol
f = plt.figure()
plt.yscale("log")


def f_err(f, data):
    for xn in data:
        yield abs(f(xn))

i, err_f_n = zip(*list(enumerate(f_err(g, n))))
j, err_f_b = zip(*list(enumerate(f_err(g, b))))

plt.plot(i, err_f_n, color="blue", label="newton")
plt.plot(j, err_f_b, color="red", label="bisect")
plt.legend()

f.savefig("speed_conv_f.pgf")

# experiments

ns = list(alg.newton(g, dgdx, 1))
bs = list(alg.bisect(g, (1, 5)))

# define the limit as the last
# iteration of newtons method.
L = ns[-1]

ns_err = (abs(xn - L) for xn in ns)
bs_err = (abs(xn - L) for xn in bs)

i, errs_newton = zip(*list(enumerate(ns_err)))
j, errs_bisect = zip(*list(enumerate(bs_err)))


# experiment for bisection
mu = 0.5
e0 = 10
es = list(alg.fpi(lambda en: mu * en, e0))

e02 = 10E-4
es2 = list(alg.fpi(lambda en: mu * en, e02))

abscisse = np.linspace(0, 100, num=len(es))

f = plt.figure()
plt.yscale("log")

plt.plot(j, errs_bisect, color="red", label="bisect")

plt.plot(
        abscisse,
        es,
        color="gray",
        label="$\\big\\{\epsilon_n\\big\\}(10)$"
)

plt.plot(
        abscisse,
        es2,
        "--",
        color="gray",
        label="$\\big\\{\epsilon_n\\big\\}\\left(10^{-4}\\right)$"
)

plt.legend()

f.savefig("exp_bisect.pgf")


# expiriment for newton method
f = plt.figure()
plt.yscale("log")
plt.ylim(10E-20, 10E0)
plt.xlim(0, 15)

print(errs_newton)

nu = 9/10
ds1 = list(alg.fpi(lambda x: nu * x**2, 1))
ds2 = list(alg.fpi(lambda x: nu * x**2, 3/4))

abscisse = np.linspace(0, 100, num=len(ds1))

plt.plot(i, errs_newton, color="blue", label="newton")

plt.plot(
        abscisse,
        ds1,
        color="gray",
        label="$\\big\\{\delta_n \\big\\} \\left( 1 \\right)$"
)

plt.plot(
        abscisse,
        ds2,
        "--",
        color="gray",
        label="$\\big\\{\delta_n \\big\\}\\left(3/4\\right)$"
)

plt.legend()

f.savefig("exp_newton.pgf")

