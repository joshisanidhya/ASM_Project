import subprocess
import sys

import pytest

pytest.importorskip("pypandoc")  # needs `uv run --group report pytest`


def test_report_builds_from_a_pipeline_run_and_refuses_synthetic_data(tmp_path):
    from headway.report import build

    outputs = tmp_path / "outputs"
    subprocess.run([sys.executable, "run.py", "--iters", "600", "--warmup", "300", "--synthetic-size", "4000",
                    "--data-dir", str(tmp_path / "no_raw_data"), "--workers", "1",
                    "--out", str(outputs)], check=True, capture_output=True)

    with pytest.raises(SystemExit, match="synthetic"):
        build(outputs, tmp_path / "report")  # a report must never be built from synthetic data

    build(outputs, tmp_path / "report", allow_synthetic=True)
    md = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "@@" not in md and (tmp_path / "report" / "report.docx").stat().st_size > 50_000
    for heading in ["Abstract", "Introduction", "Problem Statement", "Literature Review", "Data Description",
                    "Mathematical Background", "Stochastic Model", "Methodology", "Computational Implementation",
                    "Simulation Results", "Real Data Results", "Model Validation", "Discussion", "Limitations",
                    "Conclusion", "References", "Appendix"]:
        assert f"# {heading}" in md, heading
