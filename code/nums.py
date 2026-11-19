"""Number-type namespaces for sixwave.py"""
import mpmath as mp
from flint import arb
import rig

class MP:
    @staticmethod
    def root4(x): return mp.root(mp.mpf(x), 4)
    @staticmethod
    def sqrt(x): return mp.sqrt(x)
    @staticmethod
    def one(): return mp.mpf(1)
    @staticmethod
    def zero(): return mp.mpf(0)
    tanh = staticmethod(mp.tanh); atanh = staticmethod(mp.atanh); exp = staticmethod(mp.exp)

class ARB:
    @staticmethod
    def root4(x): return arb(x).root(4)
    @staticmethod
    def sqrt(x): return arb(x).sqrt()
    @staticmethod
    def one(): return arb(1)
    @staticmethod
    def zero(): return arb(0)
    @staticmethod
    def tanh(x): return arb(x).tanh()
    @staticmethod
    def atanh(x): return arb(x).atanh()
    @staticmethod
    def exp(x): return arb(x).exp()
