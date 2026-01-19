import os
from pathlib import Path

import pytest

from mhm_tools.pre.link_folder_tree import link_folder_tree


def _symlink_supported(tmp_path: Path) -> bool:
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("data", encoding="utf-8")
    try:
        dst.symlink_to(src)
    except OSError:
        return False
    return dst.is_symlink()


@pytest.mark.skipif(
    os.name == "nt", reason="Symlinks may require elevated privileges on Windows."
)
def test_links_all_files_in_tree(tmp_path: Path):
    if not _symlink_supported(tmp_path):
        pytest.skip("Symlinks not supported in this environment.")

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    (input_dir / "a").mkdir(parents=True)
    (input_dir / "a" / "file1.txt").write_text("file1", encoding="utf-8")
    (input_dir / "a" / "file2.dat").write_text("file2", encoding="utf-8")
    (input_dir / "root.txt").write_text("root", encoding="utf-8")

    link_folder_tree(input_dir=input_dir, output_dir=output_dir)

    expected_links = {
        output_dir / "a" / "file1.txt": input_dir / "a" / "file1.txt",
        output_dir / "a" / "file2.dat": input_dir / "a" / "file2.dat",
        output_dir / "root.txt": input_dir / "root.txt",
    }
    for out_file, src_file in expected_links.items():
        assert out_file.is_symlink()
        assert out_file.resolve() == src_file.resolve()


def test_file_name_filter(tmp_path: Path):
    if not _symlink_supported(tmp_path):
        pytest.skip("Symlinks not supported in this environment.")

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "keep.txt").write_text("keep", encoding="utf-8")
    (input_dir / "skip.csv").write_text("skip", encoding="utf-8")

    link_folder_tree(input_dir=input_dir, output_dir=output_dir, file_name="*.txt")

    assert (output_dir / "keep.txt").is_symlink()
    assert not (output_dir / "skip.csv").exists()


def test_overwrite_behavior(tmp_path: Path):
    if not _symlink_supported(tmp_path):
        pytest.skip("Symlinks not supported in this environment.")

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    src_file = input_dir / "data.txt"
    src_file.write_text("source", encoding="utf-8")
    out_file = output_dir / "data.txt"
    out_file.write_text("existing", encoding="utf-8")

    link_folder_tree(input_dir=input_dir, output_dir=output_dir, overwrite=False)
    assert out_file.read_text(encoding="utf-8") == "existing"
    assert not out_file.is_symlink()

    link_folder_tree(input_dir=input_dir, output_dir=output_dir, overwrite=True)
    assert out_file.is_symlink()
    assert out_file.resolve() == src_file.resolve()
