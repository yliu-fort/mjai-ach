"""Pipeline layer: the IMPALA-style executor (Ray).

RolloutWorker / Learner / ParameterHub wired by ``Runner``. Core logic
(``pipeline.rollout.RolloutWorkerCore`` etc.) is plain Python with no Ray
dependency, so unit tests run without a cluster; ``pipeline._ray`` adds the
thin ``@ray.remote`` wrappers (AGENTS.md §1 D2).

May import everything below it (league, algos, agents, games, config, utils).
"""
