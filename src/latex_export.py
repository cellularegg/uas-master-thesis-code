"""Export matplotlib figures and tables into the sibling LaTeX thesis repository."""

import locale
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
    figure.savefig(path, bbox_inches="tight")
    _print_snippet(
        f"\\includegraphics[width=\\textwidth]{{figures/{name}.pdf}}",
        name,
        caption,
        environment="figure",
    )
    return path


def save_table(
    frame: pd.DataFrame,
    name: str,
    caption: str = "",
    index: bool = True,
    **to_latex_kwargs: Any,
) -> Path:
    r"""Write a DataFrame as an ``\input``-able booktabs fragment.

    ``hrules=True`` emits ``booktabs`` rules, which the thesis preamble must load.
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
        **to_latex_kwargs: Extra arguments forwarded to
            :meth:`pandas.io.formats.style.Styler.to_latex`.

    Returns:
        Path of the written fragment, overwritten on every run.

    Raises:
        FileNotFoundError: If the thesis repository root does not exist.
    """
    path = _artifact_path("tables", f"{name}.tex")
    styler = frame.style.format(escape="latex", thousands=",", precision=2)
    for axis in (0, 1):
        styler = styler.format_index(escape="latex", axis=axis)
    if not index:
        styler = styler.hide(axis="index")
    styler.to_latex(path, hrules=True, **to_latex_kwargs)
    _print_snippet(f"\\input{{tables/{name}.tex}}", name, caption, environment="table")
    return path
