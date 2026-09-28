import json
import tempfile
import unittest
from pathlib import Path

from llm_systems_lab.artifact_audit import audit_artifacts
from llm_systems_lab.config_resolution import resolve_config, unresolved_revisions
from llm_systems_lab.manifest import create_manifest


class EvidenceGateTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
