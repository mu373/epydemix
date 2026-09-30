"""Execute selected validation notebooks with optional nbclient/nbformat.

Run from the repository root. The kernel's interpreter must have this checkout's
dependencies; --kernel selects a registered Jupyter kernel. This fails on the first
cell error and replaces stale outputs only after the entire notebook succeeds.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", help="Names relative to validation/")
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    for name in args.notebooks:
        path = directory / name
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            cell.id = hashlib.sha256(f"{name}:{index}".encode()).hexdigest()[:12]
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        start = perf_counter()
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name=args.kernel,
            resources={"metadata": {"path": str(directory)}},
        ).execute()
        nbformat.write(notebook, path)
        print(
            json.dumps(
                {
                    "notebook": name,
                    "passed": True,
                    "seconds": perf_counter() - start,
                    "source_sha256": source_hash,
                    "executed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=directory, text=True
                    ).strip(),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
