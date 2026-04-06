CLEAN.

Reviewed [global CLAUDE](C:/Users/devan/.claude/CLAUDE.md) and [project CLAUDE](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/.claude/CLAUDE.md). In [api.py#L985](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L985) and [api.py#L1277](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1277), the sync investigate responses now use `state._run_id` first and only fall back to `reasoning_trail[0].get("run_id")` if absent. Repo sweep found no remaining response-path `run_id` attribution issues of that class.

No tests run; this was a focused static review.