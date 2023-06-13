import numpy as np

def scan(f, init, xs, length=None):
    """
    A drop-in replacement for lax.scan so that you can debug w/ print
    statements, etc.
    """
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
    ys.append(y)
    return carry, np.stack(ys)