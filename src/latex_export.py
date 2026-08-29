"""Export matplotlib figures and tables into the sibling LaTeX thesis repository."""

import locale
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import MATPLOTLIB_STYLE, THESIS_DIR

plt.style.use(MATPLOTLIB_STYLE)
# Match the thesis document's typeface so exported figures read as native to it.
# The body is 11 pt; figure text one step down is correct at 1:1 scale. The locale
# is what gives tick labels thousands grouping, as in `src.plots`.
locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.formatter.use_locale": True,
    }
)


def _artifact_path(subdirectory: str, filename: str) -> Path:
    """Resolve a file path inside a subdirectory of the thesis repository.

    Args:
        subdirectory: Directory beneath the thesis root, e.g. ``"figures"``.
        filename: File name, including its extension.

    Returns:
        Path to write to; its parent directory exists.

    Raises:
        FileNotFoundError: If the thesis repository root does not exist.
    """
    if not THESIS_DIR.is_dir():
        raise FileNotFoundError(f"Thesis repository not found at {THESIS_DIR}")
    directory = THESIS_DIR / subdirectory
    directory.mkdir(exist_ok=True)
    return directory / filename


def _print_snippet(body: str, name: str, caption: str, *, environment: str) -> None:
    r"""Print the ``figure``/``table`` float to paste into a thesis chapter.

    Args:
        body: Content line placing the artifact, e.g. an ``\includegraphics``.
        name: File name stem the label is derived from.
        caption: Caption text.
        environment: LaTeX float environment, ``"figure"`` or ``"table"``.
    """
    prefix = "fig" if environment == "figure" else "tab"
    label = name.replace("_", "-")
    print(
        f"\\begin{{{environment}}}[htbp]\n"
        f"  \\centering\n"
        f"  {body}\n"
        f"  \\caption{{{caption}}}\\label{{{prefix}:{label}}}\n"
        f"\\end{{{environment}}}"
    )


def save_figure(figure: plt.Figure, name: str, caption: str = "") -> Path:
    """Write a figure as a PDF into the thesis repository's ``figures`` directory.

    Prints the LaTeX float to paste into a chapter. The figure is left open, so a
    notebook still renders it inline.

    Args:
        figure: Figure to export.
        name: File name stem, referenced from LaTeX as ``figures/{name}.pdf``.
        caption: Caption text for the printed float.

    Returns:
        Path of the written PDF.

    Raises:
        FileNotFoundError: If the thesis repository root does not exist.
    """
    path = _artifact_path("figures", f"{name}.pdf")
    # Matplotlib otherwise embeds the current time as ``/CreationDate``, which
    # makes an unchanged figure produce different PDF bytes on every run.
    figure.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    _print_snippet(
        f"\\includegraphics[width=\\textwidth]{{figures/{name}.pdf}}",
        name,
        caption,
        environment="figure",
    )
    return path


def _default_tabularx_column_format(frame: pd.DataFrame, index: bool) -> str:
    """Build a compact ``tabularx`` column specification for a frame.

    Args:
        frame: Table whose columns are being exported.
        index: Whether the DataFrame index is included in the table.

    Returns:
        A LaTeX column specification with at least one stretchable ``X`` column.
    """
    formats = ["l"] * (frame.index.nlevels if index else 0)
    formats.extend(
        "r" if pd.api.types.is_numeric_dtype(frame[column]) else "X"
        for column in frame.columns
    )
    if not formats:
        raise ValueError("Cannot export a table with no columns")
    if "X" not in formats:
        formats[0] = "X"
    return f"@{{}}{''.join(formats)}@{{}}"


def _add_latex_row_spacing(latex: str) -> str:
    r"""Insert ``\addlinespace`` between the table body rows.

    Args:
        latex: Rendered LaTeX table.

    Returns:
        The rendered table with ``\addlinespace`` between body rows.
    """
    lines = latex.splitlines()
    try:
        body_start = lines.index(r"\midrule") + 1
        body_end = lines.index(r"\bottomrule")
    except ValueError:
        return latex

    body_rows = lines[body_start:body_end]
    lines[body_start:body_end] = [
        part
        for row_number, row in enumerate(body_rows)
        for part in (([r"\addlinespace"] if row_number else []) + [row])
    ]
    return "\n".join(lines) + ("\n" if latex.endswith("\n") else "")


def save_table(
    frame: pd.DataFrame,
    name: str,
    caption: str = "",
    index: bool = True,
    addlinespace: bool = False,
    math_mode_columns: Sequence[str] = (),
    **to_latex_kwargs: Any,
) -> Path:
    r"""Write a DataFrame as an ``\input``-able ``tabularx`` booktabs fragment.

    ``hrules=True`` emits ``booktabs`` rules, and the default ``tabularx``
    environment fills the thesis text width. The thesis preamble must load both
    packages. Pass ``environment=None`` to retain pandas' plain ``tabular`` output.
    Values and both axes' labels are LaTeX-escaped, so column names such as
    ``207241-at__water_level`` and a ``25%`` index label still compile. Numbers
    are grouped with ``,`` every three digits, matching the exported figures, and
    shown to two decimals — pre-format the frame's values for anything else.

    The fragment itself carries no float wrapper; the wrapping ``table`` float is
    printed for pasting into a chapter.

    Args:
        frame: Table to export.
        name: File name stem, referenced from LaTeX as ``tables/{name}.tex``.
        caption: Caption text for the printed float.
        index: Whether to include the DataFrame index in the exported table.
        addlinespace: Whether to add ``\addlinespace`` between body rows.
        math_mode: Whether to wrap numeric cell values in LaTeX math delimiters.
        math_mode_columns: Columns containing LaTeX fragments to wrap in math
            delimiters without escaping their contents.
        environment: LaTeX table environment, defaulting to ``"tabularx"``.
        column_format: LaTeX column specification. For ``tabularx``, a suitable
            specification is generated when it is omitted.
        **to_latex_kwargs: Extra arguments forwarded to
            :meth:`pandas.io.formats.style.Styler.to_latex`.

    Returns:
        Path of the written fragment, overwritten on every run.

    Raises:
        FileNotFoundError: If the thesis repository root does not exist.
    """
    path = _artifact_path("tables", f"{name}.tex")
    math_mode = to_latex_kwargs.pop("math_mode", False)
    styler = frame.style.format(escape="latex", thousands=",", precision=2)
    for axis in (0, 1):
        styler = styler.format_index(escape="latex", axis=axis)
    if not index:
        styler = styler.hide(axis="index")
    if math_mode:
        numeric_columns = frame.select_dtypes(include="number").columns
        styler = styler.format(
            formatter=lambda value: f"${value:,.2f}$",
            subset=numeric_columns,
            escape="latex",
        )
    missing_math_columns = set(math_mode_columns).difference(frame.columns)
    if missing_math_columns:
        raise ValueError(
            "Math-mode columns are missing from the table: "
            f"{sorted(missing_math_columns)}"
        )
    if math_mode_columns:
        styler = styler.format(
            formatter=lambda value: f"${value}$",
            subset=list(math_mode_columns),
            escape=None,
        )
    environment = to_latex_kwargs.pop("environment", "tabularx")
    if environment == "tabularx":
        to_latex_kwargs.setdefault(
            "column_format", _default_tabularx_column_format(frame, index)
        )
        latex = styler.to_latex(hrules=True, **to_latex_kwargs)
        latex = latex.replace(r"\begin{tabular}", r"\begin{tabularx}{\textwidth}", 1)
        latex = latex.replace(r"\end{tabular}", r"\end{tabularx}", 1)
    else:
        latex = styler.to_latex(hrules=True, environment=environment, **to_latex_kwargs)
    if addlinespace:
        latex = _add_latex_row_spacing(latex)
    path.write_text(latex)
    _print_snippet(f"\\input{{tables/{name}.tex}}", name, caption, environment="table")
    return path
