"""Unit tests for the PS1 strategic-disclosure model.

Run from the companion/ directory, following the template's convention:

    python -m unittest discover -s tests

Every test here is a claim the paper makes. If a test fails, a sentence in the
paper is wrong -- that is the point of having them.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import strategic_disclosure as sd  # noqa: E402


class TestBrowserReplication(unittest.TestCase):
    """The Python twin must reproduce the deployed Space to the cent."""

    def test_seed_2026_draw(self):
        sp = sd.Space(seed=2026)
        self.assertEqual((sp.theta, sp.k), (2, 0.5))

    def test_boundary_case_revisable(self):
        sp = sd.Space(seed=2026, mode="memory")
        r1, r2 = sp.submit(0, 2), sp.submit(0, 2)
        self.assertEqual((r1.p, r1.mispricing, r1.payoff, r1.g), (3.0, 1.0, 1.0, 1.0))
        self.assertEqual((r2.p, r2.mispricing, r2.payoff), (2.0, 0.0, 0.0))

    def test_boundary_case_committed_control(self):
        sp = sd.Space(seed=2026, mode="static")
        self.assertEqual([sp.submit(0, 2).p for _ in range(2)], [3.0, 3.0])

    def test_typical_case_both_rules(self):
        rev = sd.Space(seed=2026, mode="memory")
        self.assertEqual([rev.submit(1, 0).p for _ in range(2)], [2.0, 3.0])
        com = sd.Space(seed=2026, mode="static")
        self.assertEqual([com.submit(1, 0).p for _ in range(2)], [2.0, 2.0])

    def test_invalid_input_is_rejected_not_clamped(self):
        with self.assertRaises(ValueError):
            sd.Space(seed=2026).submit(0, 7)
        sp = sd.Space(seed=2026)
        try:
            sp.submit(0, 7)
        except ValueError:
            pass
        self.assertEqual(len(sp.history), 0, "a rejected report must not advance state")

    def test_full_replication_assertion(self):
        self.assertTrue(sd.verify_space_replication())


class TestControlsAreAdditive(unittest.TestCase):
    """The v2 sigma/lambda controls must not move anything at their defaults."""

    def test_defaults_match_week2_version(self):
        for seed in (0, 7, 42, 2026):
            for a, m in ((0, 2), (1, 0), (1, 2)):
                explicit = sd.Space(seed=seed, mode="memory", sigma=0.5, lam=1.0)
                implicit = sd.Space(seed=seed, mode="memory")
                self.assertEqual(
                    [explicit.submit(a, m).p for _ in range(2)],
                    [implicit.submit(a, m).p for _ in range(2)],
                    f"defaults diverged at seed={seed}, a={a}, m={m}",
                )


class TestFalsePunishmentThreshold(unittest.TestCase):
    """Aaron Wang's review question: can memory mis-punish an honest firm?"""

    def _honest_round2_error(self, sigma, mode):
        sp = sd.Space(seed=2, mode=mode, sigma=sigma)
        sp.submit(1, 0)
        r = sp.submit(1, 0)
        return abs(r.p - r.v)

    def test_below_threshold_memory_helps(self):
        self.assertLess(
            self._honest_round2_error(0.5, "memory"),
            self._honest_round2_error(0.5, "static"),
            "below sigma = m_bar the revisable rule should help the honest firm",
        )

    def test_above_threshold_memory_hurts(self):
        self.assertGreater(
            self._honest_round2_error(2.0, "memory"),
            self._honest_round2_error(2.0, "static"),
            "above sigma = m_bar it should hurt -- this is the paper's threshold",
        )

    def test_analytic_share_matches_simulation(self):
        """(sigma - m_bar)/sigma, derived by hand, must match the sweep."""
        for sigma in (1.25, 1.5, 2.0, 3.0):
            got = sd.false_punishment_rate(sigma, lam=1.0)["worse_than_committed_share"]
            want = (sigma - sd.M_BAR) / sigma
            self.assertAlmostEqual(got, want, places=2, msg=f"at sigma={sigma}")

    def test_no_mis_punishment_without_noise(self):
        r = sd.false_punishment_rate(0.0, lam=1.0)
        self.assertEqual(r["worse_than_committed_share"], 0.0)


class TestPaperClaims(unittest.TestCase):
    """Headline numbers quoted in Sections 3-4."""

    def test_revisable_halves_deviation_gain(self):
        types = [(t, k) for t in sd.THETAS for k in sd.KAPPAS]
        gains = {}
        for rule in ("committed", "revisable"):
            rows = [sd.best_response(t, k, rule, 0.5, sd.LAMBDA, seed=11) for t, k in types]
            gains[rule] = sum(r["deviation_gain"] for r in rows) / len(rows)
        self.assertAlmostEqual(gains["committed"] / gains["revisable"], 2.0, places=2)

    def test_deterrence_stops_in_terminal_round(self):
        """The cheap type stops inflating in round 1 but not in round 2."""
        r = sd.best_response(2, 0.5, "revisable", 0.5, sd.LAMBDA, seed=11)
        self.assertEqual(r["m1"], 0.0, "memory should deter in the non-terminal round")
        self.assertGreater(r["m2"], 0.0, "and should fail to deter in the terminal round")

    def test_deterrence_saturates_in_lambda(self):
        types = [(t, k) for t in sd.THETAS for k in sd.KAPPAS]

        def gain(lam):
            rows = [sd.best_response(t, k, "revisable", 0.5, lam, seed=11) for t, k in types]
            return sum(r["deviation_gain"] for r in rows) / len(rows)

        self.assertAlmostEqual(gain(0.5), gain(1.0), places=6)
        self.assertAlmostEqual(gain(1.0), gain(2.0), places=6)
        self.assertGreater(gain(0.0), gain(0.5), "and it must actually fall before saturating")

    def test_vectorisation_matches_scalar_model(self):
        self.assertTrue(sd.verify_vectorisation())


if __name__ == "__main__":
    unittest.main(verbosity=2)
