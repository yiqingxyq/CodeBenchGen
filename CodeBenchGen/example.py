import os
import sys
import subprocess
import tempfile
import typing
from pathlib import Path
import rich

# Mock functionality for 'uv'
def find_uv_bin():
    return "/usr/bin/uv"  # Assume a path for the 'uv' binary for mock purposes

def uv(args: list[str], *, check: bool) -> subprocess.CompletedProcess:
    """Invoke a uv subprocess and return the result."""
    uv_path = os.fsdecode(find_uv_bin())
    result = subprocess.CompletedProcess(args, 0, stdout=b'Mock output')
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args)
    return result

import jupytext
import nbformat.v4.nbbase as nb

def code_cell(source: str, *, hidden: bool = False) -> dict:
    kwargs = {}
    if hidden:
        kwargs["metadata"] = {"jupyter": {"source_hidden": hidden}}

    return nb.new_code_cell(source, **kwargs)

def new_notebook(cells: list[dict]) -> dict:
    return nb.new_notebook(cells=cells)

def write_ipynb(nb: dict, file: Path) -> None:
    file.write_text(jupytext.writes(nb, fmt="ipynb"))

def new_notebook_with_inline_metadata(directory: Path, python: str | None = None) -> dict:
    """Create a new notebook with inline metadata."""
    with tempfile.NamedTemporaryFile(
        mode="w+",
        suffix=".py",
        delete=True,
        dir=directory,
        encoding="utf-8",
    ) as f:
        uv(
            ["init", *(["--python", python] if python else []), "--script", f.name],
            check=True,
        )
        contents = f.read().strip()
        return new_notebook(cells=[code_cell(contents, hidden=True), code_cell("")])

def get_first_non_conflicting_untitled_ipynb(directory: Path) -> Path:
    if not (directory / "Untitled.ipynb").exists():
        return directory / "Untitled.ipynb"

    for i in range(1, 100):
        if not (directory / f"Untitled{i}.ipynb").exists():
            return directory / f"Untitled{i}.ipynb"

    msg = "Could not find an available UntitledX.ipynb"
    raise ValueError(msg)

def init(path: Path | None, python: str | None, packages: typing.Sequence[str] = []) -> Path:
    """Initialize a new notebook."""
    if not path:
        path = get_first_non_conflicting_untitled_ipynb(Path.cwd())

    if path.suffix != ".ipynb":
        rich.print("File must have a `[cyan].ipynb[/cyan]` extension.", file=sys.stderr)
        sys.exit(1)

    notebook = new_notebook_with_inline_metadata(path.parent, python)
    write_ipynb(notebook, path)

    if len(packages) > 0:
        # Mocking the 'add' function from the '.add' module
        def add(path: Path, packages: typing.Sequence[str]):
            print(f"Mock add: Adding packages {packages} to {path}")

        add(path=path, packages=packages)

    return path


# Mock implementation of 'init_new_implementation' to test with


def test_init():
    temp_dir = Path("/home/user/tmp")
    # Ensure the temporary directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: No path provided
    result_1 = init(None, None)
    # Remove created file to allow init_new_implementation to reuse the name
    if result_1.exists():
        result_1.unlink()
    result_1_new = init_new_implementation(None, None)
    assert result_1 == result_1_new, "Test failed: Result paths do not match for no path provided."

    # Test 2: Path with python version
    test_path_2 = temp_dir / "Test2.ipynb"
    result_2 = init(test_path_2, "3.8")
    result_2_new = init_new_implementation(test_path_2, "3.8")
    assert result_2 == result_2_new, "Test failed: Result paths do not match for path with specific python version."

    # Test 3: Path with packages
    test_path_3 = temp_dir / "Test3.ipynb"
    result_3 = init(test_path_3, None, ["numpy", "pandas"])
    result_3_new = init_new_implementation(test_path_3, None, ["numpy", "pandas"])
    assert result_3 == result_3_new, "Test failed: Result paths do not match for path with packages."

if __name__ == "__main__":
    test_init()