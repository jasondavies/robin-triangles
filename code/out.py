"""Shared output helpers: every step writes one machine-readable JSON file to ../verification/."""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'verification')


def write(name, d):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(d, f, indent=1)
        f.write('\n')


def enc(x):
    """Exact encoding 'man*2^exp' of an exact (radius-0) arb, e.g. x.lower() or x.upper()."""
    from flint import arb
    x = arb(x)
    assert x.rad() == 0
    m, e = x.man_exp()
    return '%s*2^%s' % (m, e)


def enc_mpf(x):
    """Exact encoding 'man*2^exp' of an mpmath mpf or of a point ivmpf (e.g. an endpoint x.a, x.b of an
    mpmath.iv interval), without rounding."""
    from mpmath.libmp import to_man_exp
    if hasattr(x, '_mpi_'):
        lo, hi = x._mpi_
        assert lo == hi
        raw = lo
    else:
        raw = x._mpf_
    m, e = to_man_exp(raw)
    return '%s*2^%s' % (m, e)


def cells_dir():
    """Directory holding the (large, uncommitted) six-wave cell certificates: $CELLS_DIR or ../cells."""
    return os.environ.get('CELLS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cells'))


def cells_path(name):
    return os.path.join(cells_dir(), name)
