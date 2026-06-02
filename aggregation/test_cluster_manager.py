from aggregation.cluster_manager import ClusterManager, OfflineRunner
from aggregation.plans import AggregationPlan, AggregationRegistry


class RecordingOfflineRunner(OfflineRunner):
    def __init__(self):
        self.plan = None
        self.overrides = None

    def run(self, plan, overrides=None):
        self.plan = plan
        self.overrides = overrides or {}
        return {"version": "test-version"}


def test_time_range_override_is_runner_only():
    registry = AggregationRegistry()
    plan = AggregationPlan(
        plan_id="p1",
        collection_name="c1",
        time_window_sec=3600,
    )
    registry.register(plan)

    runner = RecordingOfflineRunner()
    manager = ClusterManager(
        engine=object(),
        registry=registry,
        offline_runner_factory=lambda _engine: runner,
        max_workers=1,
    )

    manager.run_offline(
        "p1",
        async_run=False,
        overrides={"time_range": (100.0, 200.0), "time_window_sec": 7200},
    )

    assert runner.plan.time_window_sec == 7200
    assert runner.overrides["time_range"] == (100.0, 200.0)
    assert not hasattr(runner.plan, "time_range")
