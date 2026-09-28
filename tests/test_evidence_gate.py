import json
import tempfile
import unittest
from pathlib import Path

from llm_systems_lab.artifact_audit import audit_artifacts
from llm_systems_lab.config_resolution import resolve_config, unresolved_revisions
from llm_systems_lab.manifest import create_manifest
from llm_systems_lab.report_builder import build_report


class EvidenceGateTests(unittest.TestCase):
    @staticmethod
    def _measured_artifact():
        return {
            "schema_version": "1.0",
            "metadata": {
                "experiment_id": "exp003-batching", "backend": "vllm",
                "model": "example/model", "precision": "bf16", "source": "measured",
                "hardware": "Tesla T4", "concurrency": 4,
                "environment": {"git_commit": "abc123", "nvidia_gpu": {"name": "Tesla T4"}},
                "workload": {"model_revision": "deadbeef", "backend_version": "0.9.0"},
            },
            "metrics": {"ttft_p50_ms": 12.5, "ttft_p95_ms": 21.0,
                        "request_throughput_rps": 8.25, "success_rate": 1.0},
            "traces": [{
                "request_id": "r1", "input_tokens": 8, "output_tokens": 2,
                "total_latency_ms": 30, "first_token_latency_ms": 12.5,
                "token_timestamps_ms": [12.5, 30], "succeeded": True,
                "error": None, "completion_text": "ok",
            }],
        }

    def test_config_resolver_pins_primary_and_draft_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "config.json"
            output = root / "resolved.json"
            source.write_text(json.dumps({
                "target_model": "target", "target_revision": "resolve-before-run",
                "draft_models": ["draft-a", "draft-b"],
            }), encoding="utf-8")
            resolved = resolve_config(source, output, resolver=lambda model: f"sha-{model}")
            self.assertEqual(resolved["target_revision"], "sha-target")
            self.assertEqual(resolved["draft_revisions"]["draft-a"], "sha-draft-a")
            self.assertEqual(unresolved_revisions(resolved), [])

    def test_artifact_audit_accepts_complete_measured_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({
                "files_sha256": {}, "resolved_config_sha256": "config-sha"
            }), encoding="utf-8")
            result = {
                "schema_version": "1.0",
                "metadata": {
                    "source": "measured", "hardware": "Tesla T4",
                    "environment": {"git_commit": "abc", "nvidia_gpu": {"name": "Tesla T4"}},
                    "workload": {"model_revision": "sha", "backend_version": "1.0"},
                },
                "traces": [{
                    "request_id": "r1", "input_tokens": 8, "output_tokens": 2,
                    "total_latency_ms": 10, "first_token_latency_ms": 5,
                    "token_timestamps_ms": [5, 10], "succeeded": True,
                    "error": None, "completion_text": "xx",
                }],
            }
            (root / "run.json").write_text(json.dumps(result), encoding="utf-8")
            audit = audit_artifacts(root, root / "audit.json")
            self.assertTrue(audit["passed"], audit["errors"])
            self.assertEqual(audit["request_traces"], 1)

    def test_artifact_audit_rejects_placeholders_and_missing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.json").write_text(json.dumps({
                "source": "measured", "model_revision": "resolve-before-run"
            }), encoding="utf-8")
            audit = audit_artifacts(root)
            self.assertFalse(audit["passed"])
            self.assertTrue(any("unresolved revision" in error for error in audit["errors"]))
            self.assertIn("artifact directory contains no manifest.json", audit["errors"])

    def test_manifest_rejects_unresolved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            experiment.mkdir()
            (experiment / "README.md").write_text("# Test\n", encoding="utf-8")
            config = root / "config.json"
            config.write_text(json.dumps({"model_revision": "resolve-before-run"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "still contains placeholders"):
                create_manifest(experiment, root / "manifest.json", config)

    def test_report_builder_requires_passing_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({"passed": False}), encoding="utf-8")
            limitations = root / "limitations.md"
            limitations.write_text("Limited fixture.", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not pass"):
                build_report(root, audit_path, "Report", "v1.0", "A claim.",
                             limitations, root / "report.md")

    def test_report_builder_links_only_unchanged_audited_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({
                "files_sha256": {}, "resolved_config_sha256": "config-sha"
            }), encoding="utf-8")
            run = root / "run.json"
            run.write_text(json.dumps(self._measured_artifact()), encoding="utf-8")
            audit_path = root / "audit.json"
            audit_artifacts(root, audit_path)
            limitations = root / "limitations.md"
            limitations.write_text("Measured on one T4 and one workload.", encoding="utf-8")
            output = root.parent / "report.md"
            build_report(root, audit_path, "Batching Report", "v1.0.0", "Batching improved throughput.",
                         limitations, output, "Use concurrency 4 for this workload.")
            report = output.read_text(encoding="utf-8")
            self.assertIn("Batching improved throughput.", report)
            self.assertIn("12.5", report)
            self.assertIn("8.25", report)
            self.assertIn("example/model @ deadbeef", report)
            self.assertIn("[run.json]", report)
            self.assertIn("Measured on one T4", report)

            run.write_text(json.dumps(self._measured_artifact(), indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after audit"):
                build_report(root, audit_path, "Batching Report", "v1.0.1", "A claim.",
                             limitations, output)

    def test_report_builder_rejects_json_added_after_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({
                "files_sha256": {}, "resolved_config_sha256": "config-sha"
            }), encoding="utf-8")
            (root / "run.json").write_text(json.dumps(self._measured_artifact()), encoding="utf-8")
            audit_path = root / "audit.json"
            audit_artifacts(root, audit_path)
            (root / "late.json").write_text("{}", encoding="utf-8")
            limitations = root / "limitations.md"
            limitations.write_text("Limited fixture.", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON set changed"):
                build_report(root, audit_path, "Report", "v1.0", "A claim.",
                             limitations, root.parent / "report.md")


if __name__ == "__main__":
    unittest.main()
