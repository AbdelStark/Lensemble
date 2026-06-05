"""Phase 2 experiment matrix for scaled federated LeWorldModel evidence.

The matrix is intentionally small and explicit: each row ties a public claim to
the metric, dataset/environment, controls, and falsifying result needed before
that claim can appear in a model card. It is a planning artifact, not an
experiment result.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Phase2Area = Literal["data", "deploy", "eval", "artifacts", "docs"]


class Phase2MatrixRow(BaseModel):
    """One reviewer-facing Phase 2 experiment/evidence row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issue: int = Field(ge=1)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    area: Phase2Area
    claim: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    dataset_env: str = Field(min_length=1)
    baselines: str = Field(min_length=1)
    controls: str = Field(min_length=1)
    expected_result: str = Field(min_length=1)
    falsifying_result: str = Field(min_length=1)
    artifact_gate: str = Field(min_length=1)


def default_phase2_matrix() -> tuple[Phase2MatrixRow, ...]:
    """Return the canonical Phase 2 evidence matrix for tracker #200."""

    return (
        Phase2MatrixRow(
            issue=201,
            slug="phase2-data-contract",
            area="data",
            claim="Phase 2 uses participant-local data refs that are larger than the claim-MVP smoke silos.",
            metric="load smoke window counts, dataset Merkle roots, action spec, held-out split metadata",
            dataset_env="at least two publishable LeRobot-H5 or immutable mounted silo refs",
            baselines="not applicable",
            controls="same format contract, frame skip/windowing, licensing, and residency boundary per silo",
            expected_result="all silos load through the public adapter and produce nonzero train/eval windows",
            falsifying_result="only local paths or zero-window toy data are available without a documented blocker",
            artifact_gate="dataset repos or immutable refs plus root hashes are recorded in the Phase 2 report",
        ),
        Phase2MatrixRow(
            issue=202,
            slug="phase2-gpu-federated-run",
            area="deploy",
            claim="A pinned GPU-backed HF Job can run multi-round federated LeWorldModel training and publish artifacts.",
            metric="job status, committed rounds, final global hash, per-round metric series, publication state",
            dataset_env="Phase 2 participant silos from issue 201",
            baselines="claim-MVP CPU job as the smoke lower bound",
            controls="pinned git SHA, fixed seeds, live target mode, same probe and dataset refs",
            expected_result="at least one GPU job completes and publishes checkpoint plus report artifacts",
            falsifying_result="no GPU job completes, or the report cannot be verified from the Hub",
            artifact_gate="HF job URL, checkpoint repo, report JSON, command/config hash, and final hash verify",
        ),
        Phase2MatrixRow(
            issue=206,
            slug="phase2-downstream-eval",
            area="eval",
            claim="A Phase 2 checkpoint supports downstream planning or held-out eval beyond training scalars.",
            metric="EvalReport success_rate, planning samples, time_per_action_ms, effective_dim",
            dataset_env="held-out goals or supported stable-worldmodel/synthetic env declared before evaluation",
            baselines="claim-MVP has no downstream task-scale result",
            controls="same goal set, CEM/iCEM budget, action clipping, seed, and checkpoint hash",
            expected_result="at least one checkpoint has a generated, schema-valid downstream eval report",
            falsifying_result="only prediction/SIGReg metrics exist, with no downstream report or clear blocker",
            artifact_gate="eval report is published and references the evaluated checkpoint hash",
        ),
        Phase2MatrixRow(
            issue=205,
            slug="phase2-baselines-curves",
            area="eval",
            claim="Anchored federation is compared against meaningful controls and ablations.",
            metric="curves for val_pred, val_sigreg, effective_rank, frame_drift_deg, downstream success/cost",
            dataset_env="same silos and eval task as the main Phase 2 run",
            baselines="local-only, centralized/pooled when allowed, naive FedAvg, anchored federation",
            controls="matched seeds, model size, rounds, CEM budget, and data splits",
            expected_result="generated table links each curve point to a run/config/checkpoint hash",
            falsifying_result="curves are handwritten, unmatched, or missing baseline blockers",
            artifact_gate="curve table/report is generated from run artifacts and checked into the evidence bundle",
        ),
        Phase2MatrixRow(
            issue=204,
            slug="phase2-evidence-bundle",
            area="artifacts",
            claim="The Phase 2 public story is backed by one durable evidence bundle.",
            metric="bundle schema validation, referenced artifact existence, model-card claim boundaries",
            dataset_env="all Phase 2 data, train, eval, and baseline artifacts",
            baselines="not applicable",
            controls="no raw observations/actions/embeddings in public reports; hashes and scalar metrics only",
            expected_result="bundle validates and publishes a model card plus machine-readable report",
            falsifying_result="claims depend on scattered logs or unverifiable private paths",
            artifact_gate="published checkpoint repo contains report JSON, model card, and artifact refs",
        ),
        Phase2MatrixRow(
            issue=203,
            slug="phase2-roadmap-docs",
            area="docs",
            claim="Public docs distinguish empirical Phase 2 from RFC-0006 cryptographic proof work.",
            metric="README and roadmap links to tracker #200 and child issues",
            dataset_env="documentation surface",
            baselines="claim-MVP implementation status section",
            controls="known limitations and non-claims remain visible",
            expected_result="reader can find Phase 2 scope, current evidence, and next artifact gates",
            falsifying_result="Phase 2 is ambiguous or stale claim-MVP links remain",
            artifact_gate="docs link-check and docs site build pass in CI",
        ),
    )


def render_phase2_matrix_markdown(
    rows: tuple[Phase2MatrixRow, ...] | None = None,
) -> str:
    """Render rows in the evaluation-rubric table shape."""

    resolved = rows or default_phase2_matrix()
    header = (
        "| Issue | Claim | Metric | Dataset/env | Baselines | Controls | Expected result | "
        "Falsifying result | Artifact gate |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    body = [
        "| "
        + " | ".join(
            (
                f"#{row.issue}",
                row.claim,
                row.metric,
                row.dataset_env,
                row.baselines,
                row.controls,
                row.expected_result,
                row.falsifying_result,
                row.artifact_gate,
            )
        )
        + " |"
        for row in resolved
    ]
    return "\n".join((header, *body))
