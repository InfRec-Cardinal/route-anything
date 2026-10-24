#!/usr/bin/env python3
"""Small synthetic example of the solver and certificate; not a paper experiment."""
import numpy as np
from certification import cc_lp, fixed_sequence, pvalues, price_path, ratio_opt


def main():
    # Two observable states and two actions with different quality/cost tradeoffs.
    pi = np.array([0.5, 0.5])
    value = np.array([[0.9, 0.7], [0.6, 0.8]])
    cost = np.array([[2.0, 0.5], [2.0, 0.5]])
    efficiency, _ = ratio_opt(pi, value, cost)
    path = price_path(pi, value, cost, efficiency)
    constrained = cc_lp(pi, value, cost, 0.8)
    realized_value = float(pi @ (constrained['g'] * value).sum(1))
    assert np.isclose(efficiency, 1.5)
    assert realized_value >= 0.8 - 1e-10
    assert np.all(np.diff(path['EV']) <= 1e-10)

    # A fixed order stops at its first failed certificate, even if a later item passes.
    differences = np.array([[0.2] * 2000, [-0.1] * 2000, [0.3] * 2000])
    for method in ('hoeffding', 'emp_bernstein'):
        assert fixed_sequence(pvalues(differences, 0.0)[method]) == 1
    print('Synthetic quality-floor example passed (not paper-result reproduction).')


if __name__ == '__main__':
    main()
