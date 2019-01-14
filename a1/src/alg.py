from math import cos, sin



# problem 2

def ddx(fn, dx):
    return lambda x: (fn(x + dx) - fn(x)) / dx
