"""MVP.1 deterministic evaluation harness for Irys RLM.

Pytest-native. Two runners — store mode (no LLM at all) and engine-stub mode
(canned Gemini responses). Capability-gated invariants: fixtures declare
invariant groups that only activate once the production capability they
depend on exists, so the harness can land before MVP.2/MVP.3/MVP.4.
"""
