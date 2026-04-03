CLEAN.

I don’t see any remaining HIGH or MEDIUM issues in the four requested areas on static review:

- [src/irys/matter/matter.py:250](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L250): `correct_assertion()` now deduplicates/chunks affected assertions, preloads overrides once, and batches proof-state writes inside one outer transaction. That closes the prior recompute/per-issue transaction overhead without introducing a new correctness fault in this block.
- [src/irys/matter/graph.py:489](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L489): `list_recent_for_hydration()` applies the inactive-state filter before `LIMIT`, keeps the work bounded to the filtered ID set, and returns the fields the hydration consumer actually reads, including SPO columns and `source_roles_csv`.
- [src/irys/matter/graph.py:1519](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1519): `resolve_predicate()` is matter-scoped in the `UPDATE` predicate, so foreign-matter predicate IDs are not resolved.
- [src/irys/matter/graph.py:1534](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1534): `resolve_predicate_by_description()` is a single-statement atomic `UPDATE` with matter scoping in the subquery, so it removes the prior select-then-update race.

I also checked the nearby targeted tests for these paths; they line up with the current implementations. I wasn’t able to execute `pytest` here because the shell policy blocked test commands.