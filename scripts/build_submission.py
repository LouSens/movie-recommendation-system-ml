"""Package the Dicoding submission zip.

Contents:
    sistem_rekomendasi_film.ipynb        executed notebook
    sistem_rekomendasi_film.py           exported script
    laporan_proyek_machine_learning.md   report (Indonesian)
    outputs/figures/*.png                images referenced by the report

Usage:
    python scripts/build_submission.py [--output dist/submission_movie_recommendation.zip]
"""

from __future__ import annotations

import argparse
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


def check_notebook_executed(path: Path) -> None:
    """Fail if any code cell is unexecuted or contains an error output."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
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
