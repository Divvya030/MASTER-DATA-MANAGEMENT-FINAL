"""
Clustering: converts pairwise scored matches into entity clusters using
Dedupe's own hierarchical clustering (dedupe.Dedupe.cluster()) and its own
singleton-handling static method (_add_singletons) -- not a custom
transitive-closure implementation. Every input record ends up in exactly
one cluster (singletons included), so nothing is silently dropped.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ClusterResult:
    status: str  # COMPLETED | FAILED
    clusters: list | None = None   # [{entity_id, record_ids, scores, cluster_size}, ...]
    cluster_count: int = 0
    singleton_count: int = 0
    largest_cluster_size: int = 0
    error: str | None = None


def run_clustering(raw_scores, all_record_ids: list, threshold: float = 0.5) -> ClusterResult:
    """
    raw_scores: the structured numpy Scores array from a MatchRun (loaded via
                np.load of the persisted .npy artifact) -- the SAME object
                dedupe.Dedupe.score()/StaticDedupe.score() returns.
    all_record_ids: every record ID that was part of the matched dataset,
                     so records with NO candidate pairs at all still get a
                     traceable singleton cluster.
    """
    import dedupe.clustering

    try:
        clusters_raw = list(dedupe.clustering.cluster(raw_scores, threshold=threshold))
        clusters_raw = list(dedupe.Dedupe._add_singletons(all_record_ids, clusters_raw))
    except Exception as e:  # noqa: BLE001
        return ClusterResult(status="FAILED", error=f"Clustering failed: {type(e).__name__}: {e}")

    clusters = []
    for i, (record_ids, scores) in enumerate(clusters_raw):
        clusters.append({
            "entity_id": f"ENT-{i + 1:06d}",
            "record_ids": [int(r) for r in record_ids],
            "scores": [round(float(s), 6) for s in scores],
            "cluster_size": len(record_ids),
        })

    cluster_count = len(clusters)
    singleton_count = sum(1 for c in clusters if c["cluster_size"] == 1)
    largest_cluster_size = max((c["cluster_size"] for c in clusters), default=0)

    # Traceability guarantee check: every input record must appear in
    # exactly one cluster. This is enforced by construction (cluster() +
    # _add_singletons partitions all_record_ids exactly once each), but
    # verify explicitly rather than silently trust it.
    all_clustered = [rid for c in clusters for rid in c["record_ids"]]
    if sorted(all_clustered) != sorted(set(all_record_ids)) or len(all_clustered) != len(set(all_record_ids)):
        return ClusterResult(
            status="FAILED",
            error="Traceability check failed: clustering output does not account for "
                  "every input record exactly once. Refusing to return an inconsistent result.",
        )

    return ClusterResult(
        status="COMPLETED", clusters=clusters, cluster_count=cluster_count,
        singleton_count=singleton_count, largest_cluster_size=largest_cluster_size,
    )
