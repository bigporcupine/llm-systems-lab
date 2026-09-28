import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_every_experiment_is_visible_and_documented(self):
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        index = (ROOT / "experiments" / "README.md").read_text(encoding="utf-8")
        for number in range(1, 9):
            prefix = f"exp{number:03d}_"
            matches = [path for path in (ROOT / "experiments").iterdir() if path.name.startswith(prefix)]
            self.assertEqual(len(matches), 1, prefix)
            self.assertTrue((matches[0] / "README.md").is_file())
            self.assertIn(f"{number:03d}", root_readme)
            self.assertIn(f"{number:03d}", index)

    def test_all_json_definitions_parse(self):
        for path in (ROOT / "experiments").rglob("*.json"):
            with self.subTest(path=path):
                json.loads(path.read_text(encoding="utf-8"))

    def test_documented_commands_have_cli_entrypoints(self):
        cli = (ROOT / "src" / "llm_systems_lab" / "cli.py").read_text(encoding="utf-8")
        commands = (
            "experiment-matrix", "evaluate-endpoint", "experiment-005-kv-memory",
            "experiment-005-prefix-cache", "experiment-006-speculative",
            "production-gateway", "experiment-008-load", "analyze-canary",
            "resolve-config", "create-manifest", "audit-artifacts",
            "build-report",
        )
        for command in commands:
            self.assertIn(f'"{command}"', cli)

    def test_backend_launchers_are_shell_syntax_valid(self):
        required = (
            "start_vllm_variant.sh", "start_vllm_speculative.sh",
            "start_vllm_lora.sh", "start_llamacpp_backend.sh",
            "start_tensorrt_llm_backend.sh",
        )
        for name in required:
            with self.subTest(name=name):
                path = ROOT / "scripts" / name
                self.assertTrue(path.is_file())
                subprocess.run(["bash", "-n", str(path)], check=True)


if __name__ == "__main__":
    unittest.main()
