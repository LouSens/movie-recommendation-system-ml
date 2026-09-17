"""Package the project submission zip.

Contents:
    sistem_rekomendasi_film.ipynb        executed notebook
    sistem_rekomendasi_film.py           exported script
    laporan_proyek_machine_learning.md   report (Indonesian)
    outputs/figures/*.png                images referenced by the report

Checks before packaging (the build fails if any is violated):
    * every notebook code cell has run, with no error outputs
    * the notebook is independent: it imports nothing from this repository
      (``src``, ``scripts``, ``tests``) and does not modify ``sys.path``
    * the .py contains exactly the notebook's code cells, in order, and nothing
      else besides comments (exported markdown and ``# In[n]:`` markers)
    * every image referenced by the report exists

Usage:
    python scripts/build_submission.py [--output dist/submission_movie_recommendation.zip]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "sistem_rekomendasi_film.ipynb"
SCRIPT = ROOT / "sistem_rekomendasi_film.py"
REPORT = ROOT / "laporan_proyek_machine_learning.md"
DEFAULT_OUTPUT = ROOT / "dist" / "submission_movie_recommendation.zip"

LOCAL_PACKAGES = {"src", "scripts", "tests"}
CELL_MARKER = re.compile(r"^# In\[[^\]]*\]:[ \t]*$", flags=re.MULTILINE)


def load_code_cells(path: Path) -> list[dict]:
    """Return the notebook's code cells (with outputs), in order."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def check_notebook_executed(path: Path) -> None:
    """Fail if any code cell is unexecuted or contains an error output."""
    code_cells = load_code_cells(path)
    unexecuted = [i for i, c in enumerate(code_cells) if c.get("execution_count") is None]
    errors = [
        i
        for i, c in enumerate(code_cells)
        if any(o.get("output_type") == "error" for o in c.get("outputs", []))
    ]
    if unexecuted or errors:
        raise SystemExit(
            f"Notebook not ready: unexecuted cells {unexecuted}, cells with errors {errors}. "
            "Run it top to bottom before packaging."
        )


def check_notebook_independent(path: Path) -> None:
    """Fail if the notebook imports project code or edits ``sys.path``."""
    problems: list[str] = []
    for index, cell in enumerate(load_code_cells(path)):
        source = "".join(cell["source"])
        # IPython magics and shell escapes are not valid Python; blank those lines out.
        python = "\n".join(
            "" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines()
        )
        for node in ast.walk(ast.parse(python)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    problems.append(f"cell {index}: relative import")
                modules = [node.module or ""]
            else:
                continue
            problems += [
                f"cell {index}: imports '{m}'" for m in modules if m.split(".")[0] in LOCAL_PACKAGES
            ]
        if "sys.path" in python:
            problems.append(f"cell {index}: modifies sys.path")
    if problems:
        raise SystemExit("Notebook is not self-contained:\n  " + "\n  ".join(problems))


def check_script_matches_notebook(script: Path, notebook: Path) -> None:
    """Fail unless the .py holds exactly the notebook's code cells, plus comments only."""
    cells = ["".join(c["source"]).strip("\n") for c in load_code_cells(notebook)]
    text = script.read_text(encoding="utf-8").replace("\r\n", "\n")
    header, *blocks = CELL_MARKER.split(text)

    def only_comments(chunk: str) -> bool:
        return all(not line.strip() or line.lstrip().startswith("#") for line in chunk.splitlines())

    problems: list[str] = []
    if not only_comments(header):
        problems.append("code found before the first '# In[ ]:' marker")
    if len(blocks) != len(cells):
        problems.append(f"{len(blocks)} code blocks in .py vs {len(cells)} code cells in notebook")
    for index, (code, block) in enumerate(zip(cells, blocks, strict=False)):
        body = block.strip("\n")
        if not body.startswith(code):
            problems.append(f"cell {index}: code differs from the notebook")
        elif not only_comments(body[len(code) :]):
            problems.append(f"cell {index}: extra code after the notebook cell")
    if problems:
        raise SystemExit(
            f"{script.name} does not match the notebook. Re-export it with:\n"
            f"  jupyter nbconvert --to script notebooks/{notebook.name} --output-dir .\n  "
            + "\n  ".join(problems)
        )


def referenced_images(report: Path) -> list[Path]:
    """Return local image paths referenced with ``![alt](path)`` in the report."""
    links = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", report.read_text(encoding="utf-8"))
    images = [ROOT / link for link in links if not link.startswith(("http://", "https://"))]
    missing = [str(p.relative_to(ROOT)) for p in images if not p.is_file()]
    if missing:
        raise SystemExit(f"Report references missing images: {missing}")
    return images


def build(output: Path) -> Path:
    for required in (NOTEBOOK, SCRIPT, REPORT):
        if not required.is_file():
            raise SystemExit(f"Missing required file: {required.relative_to(ROOT)}")
    check_notebook_executed(NOTEBOOK)
    check_notebook_independent(NOTEBOOK)
    check_script_matches_notebook(SCRIPT, NOTEBOOK)
    images = referenced_images(REPORT)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(NOTEBOOK, NOTEBOOK.name)
        archive.write(SCRIPT, SCRIPT.name)
        archive.write(REPORT, REPORT.name)
        for image in images:
            archive.write(image, image.relative_to(ROOT).as_posix())

    print(f"Created {output.relative_to(ROOT)} with {3 + len(images)} files:")
    with zipfile.ZipFile(output) as archive:
        for info in archive.infolist():
            print(f"  {info.filename:55s} {info.file_size / 1024:8.1f} KB")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    sys.exit(main())
