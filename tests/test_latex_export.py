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


def test_save_figure_omits_time_varying_pdf_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(latex_export, "THESIS_DIR", tmp_path)
    figure, axis = plt.subplots()
    axis.plot([1.0, 2.0], [3.0, 4.0])

    try:
        path = latex_export.save_figure(figure, "stable")
        first_bytes = path.read_bytes()

        latex_export.save_figure(figure, "stable")

        assert b"/CreationDate" not in first_bytes
        assert path.read_bytes() == first_bytes
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
    assert r"\begin{tabularx}{\textwidth}" in content
    assert r"\end{tabularx}" in content
    assert r"\toprule" in content
    assert r"\begin{table}" not in content
    # Unescaped labels would not compile: `%` starts a LaTeX comment.
    assert r"25\%" in content
    assert r"a\_\_b" in content
    assert "100,718.00" in content
    assert r"\addlinespace" not in content
    # The float that wraps the fragment is printed, not written into it.
    printed = capsys.readouterr().out
    assert r"\input{tables/demo.tex}" in printed
    assert r"\label{tab:demo}" in printed


def test_save_table_can_wrap_numbers_in_latex_math_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(latex_export, "THESIS_DIR", tmp_path)

    frame = pd.DataFrame({"count": [100718.0]})

    path = latex_export.save_table(frame, "math_mode", math_mode=True)

    assert "$100,718.00$" in path.read_text()


def test_save_table_can_omit_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(latex_export, "THESIS_DIR", tmp_path)

    frame = pd.DataFrame({"count": [100718.0]}, index=["row label"])

    path = latex_export.save_table(frame, "without_index", index=False)

    content = path.read_text()
    assert "row label" not in content
    assert "count" in content
    assert "100,718.00" in content


def test_save_table_can_add_spacing_between_body_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(latex_export, "THESIS_DIR", tmp_path)

    path = latex_export.save_table(
        pd.DataFrame({"value": [1.0, 2.0, 3.0]}),
        "with_addlinespace",
        addlinespace=True,
    )

    content = path.read_text()
    assert content.count(r"\addlinespace") == 2
    assert content.index(r"\addlinespace") < content.index(r"\bottomrule")


def test_missing_thesis_repository_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(latex_export, "THESIS_DIR", tmp_path / "absent")

    with pytest.raises(FileNotFoundError, match="absent"):
        latex_export.save_table(pd.DataFrame({"value": [1.0]}), "demo")
