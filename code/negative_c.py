"""Negative tests for the certifier of Lemma certC (they must FAIL):
 - started at M = 21/20, where the inequality (certC) is false;
 - started at M = 1, where cot(theta*) > 1 (theta* < pi/4).
Writes ../verification/negative_c.json."""
from flint import fmpq
from large_beta import certify
from out import write

r1 = certify((21, 20), fmpq(11, 10), (1, 1000), verbose=False)
r2 = certify((1, 1), fmpq(11, 10), (1, 1000), verbose=False)
write('negative_c.json', dict(start_21_20_fails=r1 is None, start_1_fails=r2 is None, ok=bool(r1 is None and r2 is None)))
