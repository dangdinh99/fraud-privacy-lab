"""Hand-computed tests for fraudpriv.drift.psi."""

import math

import numpy as np
import pytest

from fraudpriv.drift import psi


def test_identical_distributions_have_zero_psi():
    x = np.arange(1000)
    assert psi(x, x, n_bins=10) == pytest.approx(0.0)


def test_psi_hand_computed_two_bins():
    # Reference 0..99 split at the median into two halves (50% / 50%).
    # Current: 80% in the lower half, 20% in the upper half.
    ref = np.arange(100)
    cur = np.r_[np.zeros(80), np.full(20, 99)]
    expected = (0.8 - 0.5) * math.log(0.8 / 0.5) + (0.2 - 0.5) * math.log(0.2 / 0.5)
    assert psi(ref, cur, n_bins=2) == pytest.approx(expected)


def test_out_of_range_values_go_to_edge_bins_and_floor_avoids_infinity():
    ref = np.arange(100)
    cur = np.full(50, 1e6)  # all above the reference range -> last bin
    value = psi(ref, cur, n_bins=2)
    floor = 1e-4
    expected = (floor - 0.5) * math.log(floor / 0.5) + (1 - 0.5) * math.log(1 / 0.5)
    assert value == pytest.approx(expected)


def test_binary_column_merges_repeated_edges():
    ref = np.r_[np.zeros(70), np.ones(30)]
    cur = np.r_[np.zeros(40), np.ones(60)]
    expected = (0.4 - 0.7) * math.log(0.4 / 0.7) + (0.6 - 0.3) * math.log(0.6 / 0.3)
    assert psi(ref, cur, n_bins=10) == pytest.approx(expected)


def test_psi_rejects_empty_and_nan():
    with pytest.raises(ValueError):
        psi([], [1.0], n_bins=10)
    with pytest.raises(ValueError):
        psi([1.0, np.nan], [1.0], n_bins=10)
