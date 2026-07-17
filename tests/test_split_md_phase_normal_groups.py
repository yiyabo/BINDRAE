import unittest

from scripts.split_md_phase_normal_groups import (
    assign_components,
    components_from_links,
    group_labels,
    joint_components,
)


def make_grouped(system_ids):
    return {
        system_id: [
            {
                "sample_id": f"{system_id}__silver_r00",
                "audit_metrics": {
                    "confident_phase_points": 10 + index,
                    "valid_residual_points": 20 + index,
                    "active_interior_points": 100 + index,
                },
            }
        ]
        for index, system_id in enumerate(system_ids)
    }


class SplitMdPhaseNormalGroupsTest(unittest.TestCase):
    def test_joint_components_close_family_and_scaffold_links(self):
        systems = ["a", "b", "c", "d", "e", "f"]
        family_components = components_from_links(systems, [("a", "b"), ("c", "d")])
        family_labels = group_labels("fam", family_components)
        scaffold_labels = {
            "a": "s1",
            "b": "s2",
            "c": "s2",
            "d": "s3",
            "e": "s4",
            "f": "s5",
        }

        components = joint_components(systems, family_labels, scaffold_labels)

        self.assertIn(["a", "b", "c", "d"], components)
        self.assertIn(["e"], components)
        self.assertIn(["f"], components)

    def test_assignment_is_deterministic_and_keeps_components_whole(self):
        components = [
            [f"large-{index}" for index in range(8)],
            ["medium-a", "medium-b"],
            ["small-a"],
            ["small-b"],
            ["small-c"],
            ["small-d"],
        ]
        systems = [system_id for component in components for system_id in component]
        grouped = make_grouped(systems)
        fractions = {"train": 0.6, "val": 0.2, "test": 0.2}

        first = assign_components(components, grouped, fractions, seed=17)
        second = assign_components(components, grouped, fractions, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(set().union(*map(set, first.values())), set(systems))
        self.assertFalse(set(first["train"]) & set(first["val"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["val"]) & set(first["test"]))
        for component in components:
            owners = {
                split
                for split, assigned in first.items()
                if set(component) & set(assigned)
            }
            self.assertEqual(len(owners), 1)

    def test_three_way_split_requires_three_joint_groups(self):
        grouped = make_grouped(["a", "b"])
        with self.assertRaises(ValueError):
            assign_components(
                [["a"], ["b"]],
                grouped,
                {"train": 0.8, "val": 0.1, "test": 0.1},
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
