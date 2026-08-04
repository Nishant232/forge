from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.files import edit_file, list_dir, read_file, write_file


class TestReadFile:
    def test_reads_existing_file(self, project_root: Path) -> None:
        f = project_root / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        result = read_file.invoke({"path": str(f)})
        assert result == "hello world"

    def test_raises_if_not_found(self, project_root: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_file.invoke({"path": str(project_root / "missing.txt")})

    def test_raises_if_directory(self, project_root: Path) -> None:
        with pytest.raises(IsADirectoryError):
            read_file.invoke({"path": str(project_root)})

    def test_raises_outside_root(self, project_root: Path) -> None:
        with pytest.raises(PermissionError):
            read_file.invoke({"path": "/etc/passwd"})

    def test_reads_utf8_content(self, project_root: Path) -> None:
        f = project_root / "unicode.txt"
        f.write_text("café ñoño", encoding="utf-8")
        assert read_file.invoke({"path": str(f)}) == "café ñoño"


class TestWriteFile:
    def test_creates_file(self, project_root: Path) -> None:
        target = project_root / "out.txt"
        result = write_file.invoke({"path": str(target), "content": "abc"})
        assert "3 bytes" in result
        assert target.read_text() == "abc"

    def test_creates_parent_dirs(self, project_root: Path) -> None:
        target = project_root / "a" / "b" / "c.txt"
        write_file.invoke({"path": str(target), "content": "nested"})
        assert target.read_text() == "nested"

    def test_overwrites_existing(self, project_root: Path) -> None:
        target = project_root / "file.txt"
        target.write_text("old")
        write_file.invoke({"path": str(target), "content": "new"})
        assert target.read_text() == "new"

    def test_raises_outside_root(self, project_root: Path) -> None:
        with pytest.raises(PermissionError):
            write_file.invoke({"path": "/etc/passwd", "content": "evil"})

    def test_reports_byte_count(self, project_root: Path) -> None:
        target = project_root / "bytes.txt"
        result = write_file.invoke({"path": str(target), "content": "hello"})
        assert "5 bytes" in result


class TestEditFile:
    def test_replaces_unique_occurrence(self, project_root: Path) -> None:
        f = project_root / "src.py"
        f.write_text("x = 1\ny = 2\n")
        result = edit_file.invoke({"path": str(f), "old_string": "x = 1", "new_string": "x = 99"})
        assert "Replaced 1 occurrence" in result
        assert f.read_text() == "x = 99\ny = 2\n"

    def test_raises_if_not_found(self, project_root: Path) -> None:
        f = project_root / "src.py"
        f.write_text("x = 1\n")
        with pytest.raises(ValueError, match="not found"):
            edit_file.invoke({"path": str(f), "old_string": "z = 99", "new_string": "z = 0"})

    def test_raises_if_multiple_occurrences(self, project_root: Path) -> None:
        f = project_root / "src.py"
        f.write_text("foo\nfoo\n")
        with pytest.raises(ValueError, match="2 times"):
            edit_file.invoke({"path": str(f), "old_string": "foo", "new_string": "bar"})

    def test_raises_if_file_missing(self, project_root: Path) -> None:
        with pytest.raises(FileNotFoundError):
            edit_file.invoke({
                "path": str(project_root / "no.py"),
                "old_string": "x",
                "new_string": "y",
            })

    def test_raises_outside_root(self, project_root: Path) -> None:
        with pytest.raises(PermissionError):
            edit_file.invoke({"path": "/etc/hosts", "old_string": "x", "new_string": "y"})

    def test_replaces_only_first_when_count_is_one(self, project_root: Path) -> None:
        # Verify the replace is clean — no accidental extra replacements
        f = project_root / "f.py"
        f.write_text("a = 1\nb = 2\n")
        edit_file.invoke({"path": str(f), "old_string": "a = 1", "new_string": "a = 10"})
        assert "b = 2" in f.read_text()


class TestListDir:
    def test_lists_files_and_dirs(self, project_root: Path) -> None:
        (project_root / "file.py").write_text("")
        (project_root / "sub").mkdir()
        result = list_dir.invoke({"path": str(project_root)})
        assert "[F] file.py" in result
        assert "[D] sub" in result

    def test_empty_directory(self, project_root: Path) -> None:
        d = project_root / "empty"
        d.mkdir()
        result = list_dir.invoke({"path": str(d)})
        assert "empty directory" in result

    def test_raises_if_not_found(self, project_root: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list_dir.invoke({"path": str(project_root / "ghost")})

    def test_raises_if_file(self, project_root: Path) -> None:
        f = project_root / "file.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            list_dir.invoke({"path": str(f)})

    def test_raises_outside_root(self, project_root: Path) -> None:
        with pytest.raises(PermissionError):
            list_dir.invoke({"path": "/etc"})

    def test_dirs_listed_before_files(self, project_root: Path) -> None:
        (project_root / "z_file.txt").write_text("")
        (project_root / "a_dir").mkdir()
        result = list_dir.invoke({"path": str(project_root)})
        lines = result.splitlines()
        dir_line = next(i for i, l in enumerate(lines) if "[D]" in l)
        file_line = next(i for i, l in enumerate(lines) if "[F]" in l)
        assert dir_line < file_line
