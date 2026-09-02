import unittest
import os
import sys
import io
from contextlib import redirect_stderr
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conditional_main_fd_gmm import (
    evaluate_gmm_takeoff_gate,
    get_args_parser,
    resolve_gmm_cls_normalization,
    validate_gmm_takeoff_gate_args,
)


class GMMTakeoffGateTest(unittest.TestCase):
    def test_default_gate_requires_both_signals(self):
        aborted, failures = evaluate_gmm_takeoff_gate({
            "gmm_class_mean_spread_to_noise": 2.1,
            "cos_update_fd_p": -0.01,
        })
        self.assertFalse(aborted)
        self.assertEqual(failures, {"spread": False, "cosine": False})

        aborted, failures = evaluate_gmm_takeoff_gate({
            "gmm_class_mean_spread_to_noise": 1.99,
            "cos_update_fd_p": -0.01,
        })
        self.assertTrue(aborted)
        self.assertTrue(failures["spread"])

        aborted, failures = evaluate_gmm_takeoff_gate({
            "gmm_class_mean_spread_to_noise": 2.1,
            "cos_update_fd_p": 0.0,
        })
        self.assertTrue(aborted)
        self.assertTrue(failures["cosine"])

    def test_all_logic_aborts_only_when_both_fail(self):
        aborted, _ = evaluate_gmm_takeoff_gate(
            {"gmm_class_mean_spread_to_noise": 1.0,
             "cos_update_fd_p": -0.1},
            logic="all",
        )
        self.assertFalse(aborted)

        aborted, _ = evaluate_gmm_takeoff_gate(
            {"gmm_class_mean_spread_to_noise": 1.0,
             "cos_update_fd_p": 0.1},
            logic="all",
        )
        self.assertTrue(aborted)

    def test_missing_and_nonfinite_metrics_fail_closed(self):
        aborted, failures = evaluate_gmm_takeoff_gate({
            "gmm_class_mean_spread_to_noise": float("nan"),
        })
        self.assertTrue(aborted)
        self.assertEqual(failures, {"spread": True, "cosine": True})

    def test_gate_is_disabled_by_default(self):
        args = get_args_parser().parse_args([])
        self.assertEqual(args.fd_gmm_takeoff_gate_step, -1)
        validate_gmm_takeoff_gate_args(args)

    def test_cosine_window_must_exist_by_gate_step(self):
        args = Namespace(
            fd_gmm_takeoff_gate_step=100,
            fd_gmm=True,
            fd_gmm_takeoff_spread_mult=2.0,
            fd_gmm_takeoff_no_cos=False,
            fd_gmm_takeoff_cos_window=7,
            compile=False,
            epochs=1,
            steps_per_epoch=200,
            print_freq=20,
        )
        with self.assertRaisesRegex(ValueError, "diagnostic samples available"):
            validate_gmm_takeoff_gate_args(args)

        args.fd_gmm_takeoff_cos_window = 6
        validate_gmm_takeoff_gate_args(args)

    def test_class_normalization_cli_and_legacy_alias(self):
        default = get_args_parser().parse_args([])
        self.assertIsNone(default.fd_gmm_cls_normalization)
        self.assertEqual(resolve_gmm_cls_normalization(default), "self")

        fixed = get_args_parser().parse_args([
            "--fd_gmm_cls_normalization", "log_classes",
        ])
        self.assertEqual(resolve_gmm_cls_normalization(fixed), "log_classes")

        legacy = get_args_parser().parse_args(["--fd_gmm_no_normalize_cls"])
        self.assertEqual(resolve_gmm_cls_normalization(legacy), "none")

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                get_args_parser().parse_args([
                    "--fd_gmm_cls_normalization", "self",
                    "--fd_gmm_no_normalize_cls",
                ])

    def test_global_legacy_disable_and_explicit_override(self):
        parser = get_args_parser()
        legacy = parser.parse_args(["--fd_gmm_no_normalize"])
        self.assertEqual(resolve_gmm_cls_normalization(legacy), "none")

        explicit = parser.parse_args([
            "--fd_gmm_no_normalize",
            "--fd_gmm_cls_normalization", "log_classes",
        ])
        self.assertEqual(resolve_gmm_cls_normalization(explicit), "log_classes")

    def test_compile_requires_spread_only_gate(self):
        args = Namespace(
            fd_gmm_takeoff_gate_step=100,
            fd_gmm=True,
            fd_gmm_takeoff_spread_mult=2.0,
            fd_gmm_takeoff_no_cos=False,
            fd_gmm_takeoff_cos_window=5,
            compile=True,
            epochs=1,
            steps_per_epoch=200,
            print_freq=20,
        )
        with self.assertRaisesRegex(ValueError, "unavailable with --compile"):
            validate_gmm_takeoff_gate_args(args)

        args.fd_gmm_takeoff_no_cos = True
        validate_gmm_takeoff_gate_args(args)


if __name__ == "__main__":
    unittest.main()
