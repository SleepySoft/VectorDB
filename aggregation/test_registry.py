from aggregation.plans import AggregationPlan
from aggregation.registry import AggregationRegistry


def test_registry_allows_multiple_nonexclusive_plans_per_collection():
    registry = AggregationRegistry(max_plans=10, max_plans_per_collection=5)

    registry.add_plan(
        AggregationPlan(
            plan_id="agg_docs_hdbscan",
            collection_name="docs",
            name="Auto discovery",
            exclusive_collection=False,
        )
    )
    registry.add_plan(
        AggregationPlan(
            plan_id="agg_docs_kmeans",
            collection_name="docs",
            name="Fixed clusters",
            method="kmeans",
            params={"n_clusters": 8},
            exclusive_collection=False,
        )
    )

    plans = registry.find_by_collection("docs")
    assert [p.plan_id for p in plans] == ["agg_docs_hdbscan", "agg_docs_kmeans"]
    assert plans[0].name == "Auto discovery"


def test_registry_rejects_duplicate_plan_id_without_overwrite():
    registry = AggregationRegistry(max_plans=10, max_plans_per_collection=5)
    registry.add_plan(AggregationPlan(plan_id="agg_docs", collection_name="docs"))

    try:
        registry.add_plan(AggregationPlan(plan_id="agg_docs", collection_name="docs"))
    except ValueError as exc:
        assert "Plan already exists" in str(exc)
    else:
        raise AssertionError("duplicate plan_id should fail without overwrite")
