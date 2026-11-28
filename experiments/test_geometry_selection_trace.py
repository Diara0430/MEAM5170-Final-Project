"""Check that follow-up contact requires consecutive loaded frames."""
import unittest
from experiments.analyze_geometry_selection import followup_contact


class GeometryContactFollowupTest(unittest.TestCase):
    def test_transient_contact_is_not_sustained(self):
        seq=[dict(ms=t,post_groups=g) for t,g in [(20,'3'),(22,''),(24,'3'),(26,'3'),(80,'')]]
        first,stable,complete=followup_contact(seq,3,20,26)
        self.assertEqual(first,20.)
        self.assertIsNone(stable)
        self.assertTrue(complete)

    def test_three_consecutive_frames_and_truncated_followup(self):
        seq=[dict(ms=t,post_groups=g) for t,g in [(20,''),(22,'1|3'),(24,'3'),(26,'3')]]
        first,stable,complete=followup_contact(seq,3,20,26)
        self.assertEqual(first,22.)
        self.assertEqual(stable,26.)
        self.assertFalse(complete)

    def test_other_digit_does_not_count(self):
        seq=[dict(ms=t,post_groups='1') for t in (20,22,24,80)]
        self.assertEqual(followup_contact(seq,3,20,24),(None,None,True))

if __name__=='__main__':unittest.main()
