from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    from . import p11_locked_test_evaluation as evaluation
    from . import p11_locked_test_repair as repair
except ImportError:
    import p11_locked_test_evaluation as evaluation
    import p11_locked_test_repair as repair


ROOT = Path(__file__).resolve().parents[2]
FORMAL_FILES = tuple(repair.FORMAL_FILES)
PRESERVED_FILES = tuple(repair.PRESERVED_DENYLIST)
REPLACEMENT_FILES = tuple(repair.REPLACEMENT_ALLOWLIST)
EARLY_CANDIDATE_FILES = (
    "03_product_row_metrics.csv",
    "06_r1_qa_exception_sensitivity.csv",
    "07_r2_alternative_image_sensitivity.csv",
    "08_r3_g_exposure_sensitivity.csv",
    "09_r4_label_definition_robustness.csv",
)
STAGE_NAMES = (
    "starting_repository_canonical_sha",
    "frozen_authority_preflight",
    "independent_persisted_02_verification",
    "create_isolated_dry_staging",
    "independent_preserved_04_validation",
    "build_03_06_07_08_09",
    "build_05_bootstrap",
    "independent_preserved_10_validation",
    "build_11",
    "build_12",
    "staged_13_of_13_validation",
    "staged_verify_recompute_10_of_10",
    "read_only_precommit_and_ending_sha",
)
PERFORMANCE_KEY_PARTS = ("average_precision", "auroc", "recall", "lift", "ci", "score", "ranking", "threshold")


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _formal_sha(directory: Path) -> dict[str, str]:
    if not directory.is_dir() or directory.is_symlink():
        raise repair.RepairIntegrityError("canonical formal root is missing or invalid")
    entries = list(directory.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise repair.RepairIntegrityError("formal root contains aliases or non-files")
    if {item.name for item in entries} != set(FORMAL_FILES):
        raise repair.RepairIntegrityError("formal root is not the exact 00-12 set")
    return {name: repair.file_sha256(directory / name) for name in FORMAL_FILES}


def _read_artifact(path: Path, name: str) -> Any:
    if name.endswith(".csv"):
        return evaluation.read_csv(path)
    return evaluation.read_json(path)


def _base_result(canonical: Path, dry_staging: Path) -> dict[str, Any]:
    return {
        "status": "FAIL",
        "head": "",
        "canonical_root": str(canonical),
        "dry_staging_root": str(dry_staging),
        "starting_sha_status": {"status": "NOT_REACHED"},
        "authority_status": "NOT_REACHED",
        "real_02_status": "NOT_REACHED",
        "candidate_statuses": {name: "NOT_REACHED" for name in REPLACEMENT_FILES},
        "staged_validation_count": 0,
        "recompute_count": 0,
        "precommit_ready": False,
        "ending_sha_status": {"status": "NOT_REACHED"},
        "canonical_writes": 0,
        "transaction_executed": False,
        "performance_values_reported": False,
        "first_failing_stage": None,
        "failure_class": None,
        "sanitized_failure_message": None,
        "formal_state_affected": False,
        "staging_created": False,
        "stages": {f"{index:02d}": "NOT_REACHED" for index in range(1, 14)},
        "stages_reached": 0,
    }


def _sanitize_failure_message(
    exc: Exception,
) -> str:
    message = " ".join(
        str(exc).replace("\n", " ").split()
    )

    banned = (
        "score",
        "auroc",
        "auc",
        "average_precision",
        "average precision",
        "recall",
        "lift",
        "threshold",
        "point_estimate",
        "point estimate",
        "ci_lower",
        "ci_upper",
        "confidence interval",
        "ranking",
    )

    safe_parts = []

    for part in message.split(";"):
        cleaned = part.strip()

        if not cleaned:
            continue

        lowered = cleaned.lower()

        if any(
            token in lowered
            for token in banned
        ):
            continue

        safe_parts.append(cleaned)

    if safe_parts:
        return "; ".join(safe_parts)

    return "diagnostic redacted"


class _RealComponents:
    def authority_preflight(
        self,
        canonical: Path,
    ) -> dict[str, Any]:
        if canonical != Path(
            evaluation.P11_DIR
        ).resolve():
            raise repair.RepairIntegrityError(
                "real dry canonical root is not the frozen P11 root"
            )

        return {
            "status": "PASS",
            "spec": evaluation._active_maintenance_gate(),
        }

    def verify_persisted_02(
        self,
        canonical: Path,
        authority: dict[str, Any],
    ) -> dict[str, Any]:
        session = evaluation._active_maintenance_session(
            authority["spec"],
            canonical,
            live_identity=False,
            preflight_validated=True,
        )

        context = session.stage03_context()

        if (
            context.get("writes") != 0
            or context.get(
                "formal_primary_rescoring"
            ) != 0
        ):
            raise repair.RepairIntegrityError(
                "persisted-02 verification firewall mismatch"
            )

        context["spec"] = authority["spec"]
        context["session"] = session

        return {
            "status": "PASS",
            "context": context,
        }

    def build_candidates(
        self,
        canonical: Path,
        context: dict[str, Any],
        names,
    ) -> dict[str, bytes]:
        return context[
            "session"
        ].replacement_bytes(
            tuple(names),
            staged_dir=context.get("staging"),
        )

    def validate_preserved(
        self,
        name: str,
        staging: Path,
        context: dict[str, Any],
    ) -> None:
        expected = context[
            "session"
        ].reference_artifact(
            name,
            score_lane="reference",
        )

        actual = _read_artifact(
            staging / name,
            name,
        )

        repair._exact_artifact_value(
            expected,
            actual,
            "dry preserved " + name,
        )

    def validate_staged(
        self,
        staging: Path,
        context: dict[str, Any],
    ) -> int:
        staged_context = evaluation._active_context(
            context["spec"],
            staging,
            False,
            False,
        )

        result = evaluation.verify_existing_artifacts(
            staged_context,
            evaluation.reference_oracle,
        )

        if (
            result.get("writes") != 0
            or result.get("model_fits") != 0
        ):
            raise repair.RepairIntegrityError(
                "staged verify-existing firewall mismatch"
            )

        return len(
            result.get("verified_files", ())
        )

    def validate_recompute(
        self,
        staging: Path,
        context: dict[str, Any],
    ) -> int:
        staged_context = evaluation._active_context(
            context["spec"],
            staging,
            False,
            True,
        )

        result = evaluation.verify_recompute_artifacts(
            staged_context,
            evaluation.reference_oracle,
        )

        if (
            result.get("writes") != 0
            or result.get("model_fits") != 0
        ):
            raise repair.RepairIntegrityError(
                "staged verify-recompute firewall mismatch"
            )

        return len(
            result.get("verified_files", ())
        )


class _SyntheticComponents:
    def authority_preflight(self, canonical: Path) -> dict[str, Any]:
        return {"status": "PASS", "spec": {"synthetic": True}}

    def verify_persisted_02(self, canonical: Path, authority: dict[str, Any]) -> dict[str, Any]:
        expected = ("canonical:" + FORMAL_FILES[2]).encode("ascii")
        if (canonical / FORMAL_FILES[2]).read_bytes() != expected:
            raise repair.RepairIntegrityError("synthetic persisted-02 mismatch")
        return {"status": "PASS", "context": {"synthetic": True}}

    def build_candidates(
        self,
        canonical: Path,
        context: dict[str, Any],
        names,
    ) -> dict[str, bytes]:
        return {
            name: (
                "candidate:" + name
            ).encode("ascii")
            for name in names
        }

    def validate_preserved(self, name: str, staging: Path, context: dict[str, Any]) -> None:
        expected = {"value": ("canonical:" + name)}
        actual = {"value": (staging / name).read_text(encoding="ascii")}
        repair._exact_artifact_value(expected, actual, "synthetic preserved " + name)

    def validate_staged(self, staging: Path, context: dict[str, Any]) -> int:
        _formal_sha(staging)
        for name in FORMAL_FILES:
            expected = ("candidate:" + name).encode("ascii") if name in REPLACEMENT_FILES else ("canonical:" + name).encode("ascii")
            if (staging / name).read_bytes() != expected:
                raise repair.RepairIntegrityError("synthetic staged artifact mismatch")
        return len(FORMAL_FILES)

    def validate_recompute(self, staging: Path, context: dict[str, Any]) -> int:
        for name in FORMAL_FILES[1:11]:
            expected = ("candidate:" + name).encode("ascii") if name in REPLACEMENT_FILES else ("canonical:" + name).encode("ascii")
            if (staging / name).read_bytes() != expected:
                raise repair.RepairIntegrityError("synthetic recompute artifact mismatch")
        return len(FORMAL_FILES[1:11])


def _prepare_staging(canonical: Path, dry_staging: Path) -> Path:
    if dry_staging.exists() or dry_staging.is_symlink():
        raise repair.RepairIntegrityError("dry staging root already exists")
    if dry_staging.parent != canonical.parent:
        raise repair.RepairIntegrityError("dry staging root must share canonical parent")
    if not dry_staging.name.startswith(canonical.name + ".stage-dry-"):
        raise repair.RepairIntegrityError("dry staging root is not explicitly dry-only")
    parent = canonical.parent
    if (parent / (canonical.name + ".transaction.json")).exists():
        raise repair.RepairIntegrityError("real transaction journal collision")
    if list(parent.glob(canonical.name + ".backup-*")) or list(parent.glob(canonical.name + ".stage-*")):
        raise repair.RepairIntegrityError("real or stale transaction sibling exists")
    dry_staging.mkdir()
    return dry_staging


def _run_pipeline(canonical: Path, dry_staging: Path, components: Any) -> dict[str, Any]:
    result = _base_result(canonical, dry_staging)
    current_stage = 0
    stage_root: Path | None = None
    starting_sha: dict[str, str] | None = None
    candidate_bytes: dict[str, bytes] = {}

    def stage(index: int, operation):
        nonlocal current_stage
        current_stage = index
        value = operation()
        result["stages"][f"{index:02d}"] = "PASS"
        result["stages_reached"] = index
        return value

    try:
        result["head"] = _git_head()
        starting_sha = stage(1, lambda: _formal_sha(canonical))
        result["starting_sha_status"] = {"status": "PASS", "formal_count": len(starting_sha), "sha256": starting_sha}
        authority = stage(2, lambda: components.authority_preflight(canonical))
        result["authority_status"] = "PASS"
        persisted = stage(3, lambda: components.verify_persisted_02(canonical, authority))
        result["real_02_status"] = "PASS"
        context = dict(persisted["context"])
        context["spec"] = authority["spec"]
        def create_staging() -> Path:
            nonlocal stage_root
            stage_root = _prepare_staging(canonical, dry_staging)
            result["staging_created"] = True
            for name in PRESERVED_FILES:
                shutil.copyfile(canonical / name, stage_root / name)
            return stage_root

        stage_root = stage(4, create_staging)
        context["staging"] = stage_root

        stage(
            5,
            lambda: components.validate_preserved(
                "04_group_metrics.csv",
                stage_root,
                context,
            ),
        )

        def build_and_write(
            names,
        ) -> int:
            requested = tuple(names)

            batch = components.build_candidates(
                canonical,
                context,
                requested,
            )

            if (
                set(batch) != set(requested)
                or len(batch) != len(requested)
            ):
                raise repair.RepairIntegrityError(
                    "stage-scoped candidate key mismatch"
                )

            for name in requested:
                content = batch[name]

                if not isinstance(
                    content,
                    (bytes, bytearray),
                ):
                    raise repair.RepairIntegrityError(
                        "stage-scoped candidate is not bytes"
                    )

                frozen = bytes(content)

                (
                    stage_root / name
                ).write_bytes(frozen)

                candidate_bytes[name] = frozen

                result[
                    "candidate_statuses"
                ][name] = "PASS"

            return len(requested)

        stage(
            6,
            lambda: build_and_write(
                EARLY_CANDIDATE_FILES
            ),
        )

        stage(
            7,
            lambda: build_and_write(
                (
                    "05_cluster_bootstrap_uncertainty.csv",
                )
            ),
        )

        stage(
            8,
            lambda: components.validate_preserved(
                "10_r5_image_exception_diagnostics.csv",
                stage_root,
                context,
            ),
        )

        stage(
            9,
            lambda: build_and_write(
                (
                    "11_p11_summary.json",
                )
            ),
        )

        def build_provenance() -> int:
            count = build_and_write(
                (
                    "12_p11_provenance.json",
                )
            )

            repair.validate_replacement_map(
                candidate_bytes
            )

            return count

        stage(
            10,
            build_provenance,
        )

        staged_count = stage(11, lambda: components.validate_staged(stage_root, context))
        if staged_count != 13:
            raise repair.RepairIntegrityError("staged 13-of-13 validation count mismatch")
        result["staged_validation_count"] = staged_count
        recompute_count = stage(12, lambda: components.validate_recompute(stage_root, context))
        if recompute_count != 10:
            raise repair.RepairIntegrityError("staged verify-recompute count mismatch")
        result["recompute_count"] = recompute_count

        def precommit_and_end() -> None:
            assert candidate_bytes is not None
            expected_replacement_sha = {
                name: hashlib.sha256(candidate_bytes[name]).hexdigest().upper()
                for name in REPLACEMENT_FILES
            }
            expected_preserved_sha = {
                name: repair.file_sha256(canonical / name)
                for name in PRESERVED_FILES
            }
            precommit = repair.validate_transaction_precommit(
                canonical,
                stage_root,
                expected_replacement_sha=expected_replacement_sha,
                expected_preserved_sha=expected_preserved_sha,
            )
            if precommit.get("ready") is not True:
                raise repair.RepairIntegrityError("read-only transaction precommit was not ready")
            result["precommit_ready"] = True
            ending = _formal_sha(canonical)
            result["ending_sha_status"] = {"status": "PASS" if ending == starting_sha else "FAIL", "formal_count": len(ending), "sha256": ending}
            if ending != starting_sha:
                raise repair.RepairIntegrityError("ending canonical SHA mismatch")

        stage(13, precommit_and_end)
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["first_failing_stage"] = f"{current_stage:02d}" if current_stage else None
        result["failure_class"] = type(exc).__name__
        result["sanitized_failure_message"] = (
            _sanitize_failure_message(exc)
        )
        if starting_sha is not None and current_stage >= 2:
            try:
                ending = _formal_sha(canonical)
                result["ending_sha_status"] = {"status": "PASS" if ending == starting_sha else "FAIL", "formal_count": len(ending), "sha256": ending}
            except Exception as ending_exc:
                result["ending_sha_status"] = {"status": "FAIL", "failure_class": type(ending_exc).__name__}
    finally:
        if stage_root is not None and stage_root.exists():
            shutil.rmtree(stage_root)
    result["stages_reached"] = sum(value == "PASS" for value in result["stages"].values())
    return result


def run_read_only_dry_validation(canonical_root: Path | str, dry_staging_root: Path | str) -> dict[str, Any]:
    """Run the fixed production dry-validation orchestration without a transaction."""
    canonical = Path(canonical_root).resolve()
    dry_staging = Path(dry_staging_root).resolve()
    return _run_pipeline(canonical, dry_staging, _RealComponents())


def run_synthetic_dry_validation(canonical_root: Path | str, dry_staging_root: Path | str) -> dict[str, Any]:
    """Exercise the exact stage orchestration against deterministic test inputs."""
    return _run_pipeline(Path(canonical_root).resolve(), Path(dry_staging_root).resolve(), _SyntheticComponents())


def _validate_report_shape(value: Any) -> None:
    if not isinstance(value, dict):
        raise repair.RepairIntegrityError("dry-validation report must be an object")
    for key, child in value.items():
        lowered = str(key).lower()
        if any(part in lowered for part in PERFORMANCE_KEY_PARTS):
            raise repair.RepairIntegrityError("performance field leaked into dry-validation report")
        if isinstance(child, dict):
            _validate_report_shape(child)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, dict):
                    _validate_report_shape(item)


def serialize_result(result: dict[str, Any]) -> str:
    _validate_report_shape(result)
    return json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P11 read-only dry-validation harness")
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--dry-staging-root", required=True)
    args = parser.parse_args(argv)
    result = run_read_only_dry_validation(args.canonical_root, args.dry_staging_root)
    print(serialize_result(result), end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
