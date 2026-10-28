"""Чтение логов nano-vLLM из распакованной папки runs или ZIP-архива."""

import zipfile
from pathlib import Path


class DirectoryArchive:
    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"Папка не найдена: {self.root}")
        self._names = [
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file()
        ]

    def namelist(self):
        return self._names

    def read(self, name):
        return (self.root / name).read_bytes()

    def open(self, name):
        return (self.root / name).open("rb")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def open_dataset(path):
    path = Path(path)
    if path.is_dir():
        return DirectoryArchive(path)
    if path.is_file() and zipfile.is_zipfile(path):
        return zipfile.ZipFile(path)
    raise ValueError(f"Ожидалась папка runs или ZIP-архив с прогонами: {path}")
