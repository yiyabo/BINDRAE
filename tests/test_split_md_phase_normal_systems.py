import unittest

from scripts.split_md_phase_normal_systems import choose_split, group_records


class SplitMdPhaseNormalSystemsTest(unittest.TestCase):
    def test_replica_groups_never_cross_split(self):
        base_ids = [f"system-{index}" for index in range(8)]
        records = []
        for index, base_id in enumerate(base_ids):
            for replica in range(1 + index % 3):
                records.append(
                    {
                        "sample_id": f"{base_id}__silver_r{replica:02d}",
                        "audit_metrics": {
                            "confident_phase_points": 10 + index,
                            "valid_residual_points": 20 + replica,
                            "active_interior_points": 100 + index,
                        },
                    }
                )
        grouped = group_records(records, base_ids)
        train, val = choose_split(grouped, val_systems=2, seed=7)

        self.assertEqual(len(train), 6)
        self.assertEqual(len(val), 2)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(base_ids))
        for base_id, rows in grouped.items():
            owner = "train" if base_id in train else "val"
            self.assertTrue(all(row["sample_id"].startswith(base_id) for row in rows))
            self.assertIn(owner, {"train", "val"})

    def test_unknown_replica_base_fails(self):
        with self.assertRaises(ValueError):
            group_records([{"sample_id": "unknown__silver_r00"}], ["known"])


if __name__ == "__main__":
    unittest.main()
