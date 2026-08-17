from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src import latex_export


def test_save_figure_writes_pdf_and_prints_its_float(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(latex_export, "THESIS_DIR", tmp_path)
    figure, axis = plt.subplots()
    axis.plot([1.0, 2.0], [3.0, 4.0])

    try:
        path = latex_export.save_figure(figure, "water_level", caption="A caption.")

        assert path == tmp_path / "figures" / "water_level.pdf"
        assert path.stat().st_size > 0
        assert capsys.readouterr().out == (
            "\\begin{figure}[htbp]\n"
            "  \\centering\n"
            "  \\includegraphics[width=\\textwidth]{figures/water_level.pdf}\n"
            "  \\caption{A caption.}\\label{fig:water-level}\n"
            "\\end{figure}\n"
        )
    finally:
        plt.close(figure)


def test_save_table_writes_booktabs_fragment_without_float_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(latex_export, "THESIS_DIR", tmp_path)

    frame = pd.DataFrame({"a__b": [100718.0, 2.0]}, index=["25%", "50%"])

    path = latex_export.save_table(frame, "demo")

    content = path.read_text()
    assert path == tmp_path / "tables" / "demo.tex"
    assert r"\toprule" in content
    assert r"\begin{table}" not in content
    # Unescaped labels would not compile: `%` starts a LaTeX comment.
    assert r"25\%" in content
    assert r"a\_\_b" in content
    assert "100,718.00" in content
    # The float that wraps the fragment is printed, not written into it.
    printed = capsys.readouterr().out
    assert r"\input{tables/demo.tex}" in printed
    assert r"\label{tab:demo}" in printed


def test_missing_thesis_repository_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(latex_export, "THESIS_DIR", tmp_path / "absent")

    with pytest.raises(FileNotFoundError, match="absent"):
        latex_export.save_table(pd.DataFrame({"value": [1.0]}), "demo")
