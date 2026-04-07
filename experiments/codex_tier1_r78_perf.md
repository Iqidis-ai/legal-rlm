CLEAN

I did not find a material performance regression in commit `b3e6237`. The changes are limited to constant-time control-flow checks, log wording, and a tighter `run_session` update predicate that still keys off the run ID rather than introducing any new scan-heavy or repeated expensive work.