import unittest

from compute_repr_stats import validate_class_ids
from conditional_main_fd_mae import validate_train_class_ids


class ClassSubsetValidationTest(unittest.TestCase):
    def test_none_keeps_full_class_distribution(self):
        self.assertIsNone(validate_class_ids(None, 1000))
        self.assertIsNone(validate_train_class_ids(None, 1000))

    def test_valid_subset_is_preserved(self):
        class_ids = [0, 207, 979]
        self.assertEqual(validate_class_ids(class_ids, 1000), class_ids)
        self.assertEqual(validate_train_class_ids(class_ids, 1000), class_ids)

    def test_duplicate_ids_are_rejected(self):
        for validator in (validate_class_ids, validate_train_class_ids):
            with self.subTest(validator=validator.__name__):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    validator([9, 9], 1000)

    def test_out_of_range_ids_are_rejected(self):
        for validator in (validate_class_ids, validate_train_class_ids):
            with self.subTest(validator=validator.__name__):
                with self.assertRaisesRegex(ValueError, "outside"):
                    validator([0, 1000], 1000)


if __name__ == "__main__":
    unittest.main()
