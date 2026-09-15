"""Exercise diagnostic scheduling without importing the DMLab runtime."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

SOURCE = Path(__file__).resolve().parents[1] / "dmlab" / "custom_learner.py"
if not SOURCE.exists():
    SOURCE = Path(__file__).with_name("custom_learner.py")
TREE = ast.parse(SOURCE.read_text())
LEARNER = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "DistanceLearnerReward")


def method(name):
    return next(n for n in LEARNER.body if isinstance(n, ast.FunctionDef) and n.name == name)


def execute(node, namespace):
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)


class GoalDiagnosticScheduleTest(unittest.TestCase):
    def test_only_selected_summary_minibatch_requests_diagnostics(self):
        call = next(
            n
            for n in ast.walk(method("_train"))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_calculate_losses"
        )
        expr = next(k.value for k in call.keywords if k.arg == "record_goal_diagnostics")
        code = compile(ast.Expression(expr), str(SOURCE), "eval")
        for with_summaries in (False, True):
            requested = [
                (epoch, batch_num)
                for epoch in range(3)
                for batch_num in range(4)
                if eval(
                    code,
                    dict(
                        with_summaries=with_summaries,
                        epoch=epoch,
                        batch_num=batch_num,
                        summaries_epoch=1,
                        summaries_batch=2,
                    ),
                )
            ]
            self.assertEqual(requested, [(1, 2)] if with_summaries else [])

    def diagnostic(self, enabled):
        gate = next(
            n
            for n in ast.walk(method("_calculate_losses"))
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "record_goal_diagnostics"
        )
        original = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        alternate = original.detach().flip(0)
        tail = Mock(return_value=dict(action_logits=alternate, values=torch.tensor([1.0, 3.0])))
        owner = SimpleNamespace(
            actor_critic=SimpleNamespace(forward_tail=tail),
            _uses_policy_graph=lambda: True,
            _behavior_targets_from_states=lambda states: torch.eye(2),
            _with_worker_target=lambda core, target: core,
        )
        helper_namespace = dict(
            torch=torch,
            categorical_action_total_variation=lambda a, b: (a.softmax(-1) - b.softmax(-1)).abs().sum(-1) / 2,
        )
        execute(method("_record_goal_condition_diagnostics"), helper_namespace)
        owner._record_goal_condition_diagnostics = lambda *args: helper_namespace["_record_goal_condition_diagnostics"](
            owner, *args
        )
        stats = {}
        execute(
            gate,
            dict(
                torch=torch,
                self=owner,
                record_goal_diagnostics=enabled,
                additional_stats=stats,
                mb=SimpleNamespace(rnn_states=None),
                outputs=SimpleNamespace(
                    core_outputs=original, result=dict(action_logits=original, values=torch.tensor([0.0, 1.0]))
                ),
                categorical_action_total_variation=lambda a, b: (a.softmax(-1) - b.softmax(-1)).abs().sum(-1) / 2,
            ),
        )
        return stats, tail, original

    def test_skipped_diagnostic_does_not_call_tail_or_emit_measurements(self):
        stats, tail, _ = self.diagnostic(False)
        tail.assert_not_called()
        self.assertEqual(stats, {})

    def test_selected_diagnostic_preserves_values_without_gradients(self):
        stats, tail, original = self.diagnostic(True)
        tail.assert_called_once()
        self.assertAlmostEqual(stats["goal_condition_action_sensitivity"].item(), 2.0)
        self.assertAlmostEqual(stats["goal_condition_action_probability_tv"].item(), 0.76159416)
        self.assertAlmostEqual(stats["goal_condition_value_span"].item(), 1.5)
        self.assertTrue(all(not value.requires_grad for value in stats.values()))
        self.assertIsNone(original.grad)

    def test_forced_summary_omits_unmeasured_diagnostics(self):
        loop = next(
            n
            for n in ast.walk(method("_record_summaries"))
            if isinstance(n, ast.For)
            and isinstance(n.iter, ast.Tuple)
            and any(isinstance(e, ast.Constant) and e.value == "action_probability_tv" for e in n.iter.elts)
        )
        for additional in ({}, {"goal_condition_action_sensitivity": torch.tensor(0.04)}):
            stats = {}
            execute(loop, dict(stats=stats, var=SimpleNamespace(additional_stats=additional)))
            self.assertEqual(set(stats), {"hrl_" + key for key in additional})


if __name__ == "__main__":
    unittest.main()
