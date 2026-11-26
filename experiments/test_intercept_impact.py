import unittest
from unittest.mock import Mock

from controllers.intercept_impact import Impact, InterceptImpact, prefer_alternative


class TestImpactChoice(unittest.TestCase):
    def test_respect_force_and_joint_margin(self):
        first = Impact(3.,30.,.30)
        self.assertFalse(prefer_alternative(first, Impact(4.1,20.,.20)))
        self.assertFalse(prefer_alternative(first, Impact(2.,21.,.32)))
        self.assertTrue(prefer_alternative(first, Impact(3.9,28.,.29)))
        self.assertTrue(prefer_alternative(Impact(8.,31.,.31), Impact(6.,30.,.30)))

    def test_expired_budget_keeps_baseline(self):
        selector = InterceptImpact.__new__(InterceptImpact)
        selector.attempts = selector.timeouts = 0
        first, second = Mock(intercept_time=.4), Mock(intercept_time=.39)
        self.assertIs(selector.choose(None,.1,first,second,-1.), first)
        self.assertEqual(selector.timeouts,1)


if __name__=="__main__":
    unittest.main()
