import argparse
import unittest

import gymnasium as gym

from sample_factory.model.action_parameterization import ActionParameterizationDefault
from sf_working_directories.IntrMotiv.dmlab.custom_params import add_hipposlam_env_args
from sf_working_directories.IntrMotiv.dmlab.dmlab_gym import ACTION_SET, NAVIGATION_ACTION_SET, REDUCED_ACTION_SET


class NavigationActionSetTests(unittest.TestCase):
    def test_navigation_action_vectors_are_exact_and_exclude_fire(self):
        self.assertEqual(
            NAVIGATION_ACTION_SET,
            (
                (0, 0, 0, 1, 0, 0, 0),
                (0, 0, 0, -1, 0, 0, 0),
                (0, 0, -1, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0, 0),
                (-20, 0, 0, 0, 0, 0, 0),
                (20, 0, 0, 0, 0, 0, 0),
                (-20, 0, 0, 1, 0, 0, 0),
                (20, 0, 0, 1, 0, 0, 0),
            ),
        )
        self.assertTrue(all(vector[4] == 0 for vector in NAVIGATION_ACTION_SET))

    def test_historical_sets_are_unchanged(self):
        self.assertEqual(len(ACTION_SET), 9)
        self.assertEqual(ACTION_SET[-1], (0, 0, 0, 0, 1, 0, 0))
        self.assertEqual(len(REDUCED_ACTION_SET), 5)

    def test_navigation_switch_is_persistable_configuration(self):
        parser = argparse.ArgumentParser()
        add_hipposlam_env_args(parser)
        cfg = parser.parse_args(["--dmlab_navigation_action_set=True"])
        self.assertTrue(cfg.dmlab_navigation_action_set)
        self.assertFalse(cfg.dmlab_reduced_action_set)
        cfg = parser.parse_args(["--dmlab_navigation_action_set=False", "--dmlab_reduced_action_set=False"])
        self.assertFalse(cfg.dmlab_navigation_action_set)
        self.assertFalse(cfg.dmlab_reduced_action_set)

    def test_actor_head_has_eight_outputs_and_rejects_reduced_head(self):
        navigation_head = ActionParameterizationDefault(None, 32, gym.spaces.Discrete(len(NAVIGATION_ACTION_SET)))
        reduced_head = ActionParameterizationDefault(None, 32, gym.spaces.Discrete(len(REDUCED_ACTION_SET)))
        self.assertEqual(navigation_head.distribution_linear.out_features, 8)
        with self.assertRaises(RuntimeError):
            navigation_head.load_state_dict(reduced_head.state_dict())


if __name__ == "__main__":
    unittest.main()
