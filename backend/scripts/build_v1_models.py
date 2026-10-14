"""Rebuild a bundled V1 model:  python scripts/build_v1_models.py supplier <Supplier_1_.csv>
                                python scripts/build_v1_models.py customer <customer.csv>
Writes backend/models_bundled/<domain>/v1/{model.settings,training.json,metadata.json,evaluation.json}.
Offline tooling: the running app NEVER trains V1; it only loads these artifacts."""
import json, sys, datetime as dt
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from scripts.v1_build import specs  # noqa: E402


def main(domain, csv, scratch="/tmp/v1_build_scratch"):
    spec = {"supplier": specs.SUPPLIER_V1, "customer": specs.CUSTOMER_V1}[domain]
    if domain == "supplier":
        from scripts.v1_build import supplier_build as B
        out = B.run(csv, Path(scratch) / domain, log=lambda m: print(m, flush=True))
    else:
        import os
        if os.environ.get("CUSTOMER_V11_EXPERIMENT") == "1":
            # EXPERIMENTAL, NOT SHIPPED: the retrain did not beat V1.0 (see V1_EVALUATION_REPORT.md section 3).
            from scripts.v1_build import customer_v11_build as B
            labels = HERE / "models_bundled" / "customer" / "v1" / "assistant_reviewed_labels.json"
            out = B.run(csv, str(labels), Path(scratch) / domain, log=lambda m: print(m, flush=True))
        else:
            from scripts.v1_build import customer_build as B
            out = B.run(csv, Path(scratch) / domain, log=lambda m: print(m, flush=True))
    res = out["results"]; meta = res["meta"]
    from app.modules.dedupe_service.model_store import dependency_versions
    d = HERE / "models_bundled" / domain / "v1"; d.mkdir(parents=True, exist_ok=True)
    (d / "model.settings").write_bytes(out["settings"]); (d / "training.json").write_bytes(out["training"])
    md = {"model_format_version": 1, "training_run_id": spec["model_id"], "domain": domain, "model_version": spec["version"],
          "source_dataset_id": "BUNDLED-V1", "er_config_version": "bundled-v1", "match_fields": [f for f, _ in spec["fields"]],
          "field_types": dict(spec["fields"]), "identifier_fields": spec["identifier_fields"], "name_fields": spec["name_fields"],
          "threshold": meta["threshold"], "review_threshold": meta["review_threshold"], "threshold_calibration": meta["threshold_calibration"],
          "source_mapping": spec["training_mapping"], "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
          "dependency_versions": dependency_versions(), "training_summary": meta, "evaluation_summary": res.get("headline"),
          "provenance": "OFFLINE_BUILD scripts/build_v1_models.py"}
    (d / "metadata.json").write_text(json.dumps(md, indent=2, default=str))
    (d / "evaluation.json").write_text(json.dumps(res, indent=2, default=str))
    print("evaluation + artifacts written")
    print("wrote", d)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
