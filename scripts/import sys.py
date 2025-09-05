import sys
import json
import tempfile
from pathlib import Path
import pytest
from scripts.cloud_training import CloudTrainingManager
import pytest

# scripts/test_cloud_training.py




class TestCloudTrainingManagerDataset:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "enhanced_dataset.jsonl"
        self.manager = CloudTrainingManager(
            dataset_path=str(self.dataset_path),
            output_dir=str(Path(self.temp_dir.name) / "output"),
            platform="colab"
        )
        # Avoid logging to stdout during tests
        self.manager.logger.disabled = True

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_create_enhanced_dataset_creates_valid_jsonl(self):
        self.manager.create_enhanced_dataset()
        assert self.dataset_path.exists()
        lines = list(self.dataset_path.read_text(encoding="utf-8").splitlines())
        assert len(lines) >= 4  # At least 2 code gen + 2 code analysis
        for line in lines:
            obj = json.loads(line)
            assert "instruction" in obj
            assert "output" in obj
            assert isinstance(obj["instruction"], str) and obj["instruction"].strip()
            assert isinstance(obj["output"], str) and obj["output"].strip()
        # Check at least one code generation and one code analysis example
        instructions = [json.loads(l)["instruction"].lower() for l in lines]
        assert any("détection d'intrusion" in i or "framework de test" in i for i in instructions)
        assert any("analyser ce projet django" in i for i in instructions)

    def test_create_enhanced_dataset_overwrites_existing_file(self):
        # Write dummy/corrupted content
        self.dataset_path.write_text("corrupted\n", encoding="utf-8")
        self.manager.create_enhanced_dataset()
        lines = list(self.dataset_path.read_text(encoding="utf-8").splitlines())
        assert len(lines) >= 4
        for line in lines:
            obj = json.loads(line)
            assert "instruction" in obj
            assert "output" in obj

    def test_create_enhanced_dataset_is_idempotent(self):
        self.manager.create_enhanced_dataset()
        content1 = self.dataset_path.read_text(encoding="utf-8")
        self.manager.create_enhanced_dataset()
        content2 = self.dataset_path.read_text(encoding="utf-8")
        assert content1 == content2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])