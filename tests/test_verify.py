from pathlib import Path

from useeg.config import load_config
from useeg.verify import verify_repository

ROOT = Path(__file__).resolve().parents[1]


def test_publication_sources_and_protocol_verify() -> None:
    config = load_config(ROOT / "configs" / "paper.yaml")
    report = verify_repository(config, write_report=False)
    assert report["reported_values"]["actual"]["erp_average_auc"] == 91.80
    assert len(report["source_hashes"]) == len(config["verification"]["compare_publication_tables"]) + len(
        config["verification"]["compare_figure_targets"]
    ) + 1
