"""Offline tests for the SQL coarse-shape mapper, on hand-written SQL with known shapes.

    .venv/bin/python -m unittest bird_graph_rl/sql_shapes_test.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sql_shapes import coarse_key, shape_of  # noqa: E402


class ShapeTest(unittest.TestCase):
    def check(self, sql: str, **expected: object) -> dict:
        shape = shape_of(sql)
        for key, value in expected.items():
            self.assertEqual(shape[key], value, f"{key} for: {sql}")
        return shape

    def test_plain_lookup(self) -> None:
        self.check("SELECT name FROM users WHERE id = 5", hops=0, aggregation="none", grouping=False,
                   ordering="none", extras=[], filters=["equality"], return_columns=1)

    def test_join_count(self) -> None:
        self.check("SELECT COUNT(T1.id) FROM posts AS T1 INNER JOIN users AS T2 ON T1.owner = T2.id WHERE T2.name = 'a'",
                   hops=1, aggregation="count", extras=[])

    def test_count_distinct_and_three_tables(self) -> None:
        self.check("SELECT COUNT(DISTINCT T1.id) FROM a AS T1 JOIN b AS T2 ON T1.x = T2.x JOIN c AS T3 ON T2.y = T3.y",
                   hops=2, aggregation="count_distinct")

    def test_comma_join_counts_as_hop(self) -> None:
        self.check("SELECT a.x FROM a, b WHERE a.id = b.id", hops=1)

    def test_group_by_having(self) -> None:
        self.check("SELECT owner, COUNT(*) FROM posts GROUP BY owner HAVING COUNT(*) > 3",
                   grouping=True, aggregation="count", extras=["having"])

    def test_orderings(self) -> None:
        self.check("SELECT name FROM users ORDER BY rep DESC LIMIT 1", ordering="argmax_argmin")
        self.check("SELECT name FROM users ORDER BY rep DESC LIMIT 5", ordering="top_k")
        self.check("SELECT name FROM users ORDER BY rep", ordering="order_by")
        self.check("SELECT name FROM users LIMIT 3", ordering="limit_only")
        self.check("SELECT name FROM users ORDER BY rep DESC LIMIT 2, 1", ordering="nth_ranked")
        self.check("SELECT name FROM users ORDER BY rep DESC LIMIT 1 OFFSET 2", ordering="nth_ranked")

    def test_argmax_via_subquery(self) -> None:
        shape = self.check("SELECT name FROM users WHERE rep = (SELECT MAX(rep) FROM users)", hops=1, ordering="none")
        self.assertIn("argmax_via_subquery", shape["extras"])
        self.assertIn("subquery", shape["extras"])

    def test_existence_and_negation(self) -> None:
        shape = shape_of("SELECT name FROM users WHERE id NOT IN (SELECT owner FROM posts)")
        self.assertEqual(shape["extras"], ["existence", "negation", "subquery"])
        self.assertIn("existence", shape_of("SELECT 1 FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.x = a.x)")["extras"])
        self.assertEqual(shape_of("SELECT name FROM users WHERE age != 3")["extras"], ["negation"])

    def test_in_list_is_a_filter_not_existence(self) -> None:
        shape = self.check("SELECT name FROM users WHERE id IN (1, 2, 3)", extras=[])
        self.assertEqual(shape["filters"], ["in_list"])

    def test_ratio_with_conditional_aggregate(self) -> None:
        shape = shape_of("SELECT CAST(SUM(CASE WHEN score > 5 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(id) FROM posts")
        self.assertEqual(shape["extras"], ["conditional_aggregate", "ratio"])
        self.assertEqual(shape["aggregation"], "multiple")

    def test_iif_counts_as_conditional_aggregate(self) -> None:
        self.assertIn("conditional_aggregate", shape_of("SELECT SUM(IIF(x = 1, 1, 0)) FROM t")["extras"])

    def test_difference_of_aggregates_and_two_entities(self) -> None:
        shape = shape_of("SELECT SUM(CASE WHEN name = 'A' THEN views ELSE 0 END) - SUM(CASE WHEN name = 'B' THEN views ELSE 0 END) FROM users")
        self.assertEqual(shape["extras"], ["comparison_two_entities", "conditional_aggregate", "difference_of_aggregates"])

    def test_subtraction_of_columns_is_not_a_difference_of_aggregates(self) -> None:
        self.assertEqual(shape_of("SELECT a - b FROM t")["extras"], [])

    def test_set_operation_is_not_called_a_subquery(self) -> None:
        shape = shape_of("SELECT x FROM a UNION SELECT x FROM b")
        self.assertEqual(shape["extras"], ["set_operation"])
        self.assertEqual(shape["hops"], 1)

    def test_distinct_filters_and_dates(self) -> None:
        shape = shape_of("SELECT DISTINCT name FROM users WHERE age BETWEEN 1 AND 9 AND bio LIKE '%x%' AND loc IS NULL AND strftime('%Y', d) = '2010'")
        self.assertEqual(shape["extras"], ["distinct"])
        self.assertEqual(shape["filters"], ["date_function", "equality", "like", "null_test", "range"])

    def test_backtick_identifiers_and_hop_bucket(self) -> None:
        shape = shape_of("SELECT T1.`first name` FROM a AS T1 JOIN b AS T2 ON T1.i = T2.i JOIN c AS T3 ON 1=1 JOIN d AS T4 ON 1=1 JOIN e AS T5 ON 1=1")
        self.assertEqual((shape["hops"], shape["hops_bucket"]), (4, "4+"))

    def test_coarse_key_is_stable_and_parameter_free(self) -> None:
        a = coarse_key(shape_of("SELECT COUNT(*) FROM p JOIN u ON p.o = u.i WHERE u.name = 'x'"))
        b = coarse_key(shape_of("SELECT COUNT(*) FROM p JOIN u ON p.o = u.i WHERE u.name = 'completely different'"))
        self.assertEqual(a, b)
        self.assertEqual(a, "hops=1|agg=count|group=0|order=none|extras=-")

    def test_unparseable_raises(self) -> None:
        with self.assertRaises(Exception):
            shape_of("SELEC name FRM users WHERE")


if __name__ == "__main__":
    unittest.main()
