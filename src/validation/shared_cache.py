"""Checksummed immutable bundles, serialized per key and published atomically."""
from contextlib import contextmanager
import fcntl
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from pipeline_runtime import read_json, write_json


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class SharedCache:
    def __init__(self, context, kind, key):
        self.root = context.run_root.parent.parent / 'shared_cache' / 'v2' / kind
        self.key = key
        self.path = self.root / key

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / f'.{self.key}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def valid(self):
        try:
            manifest = read_json(self.path / 'cache.json')
            files = manifest['files']
            if not isinstance(files, dict) or not isinstance(manifest.get('origin'), dict):
                return False
            return (manifest['key'] == self.key and bool(files)
                    and all(Path(name).name == name and checksum(self.path / name) == digest
                            for name, digest in files.items()))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False

    def restore(self, destination):
        with self.locked():
            if not self.valid():
                return None
            manifest = read_json(self.path / 'cache.json')
            destination.mkdir(parents=True, exist_ok=True)
            for name in manifest['files']:
                shutil.copy2(self.path / name, destination / name)
            return manifest

    def publish(self, source, names, *, origin, replace_corrupt=True):
        with self.locked():
            if self.valid() or (self.path.exists() and not replace_corrupt):
                return
            with tempfile.TemporaryDirectory(dir=self.root, prefix=f'.{self.key}.') as temp:
                staging = Path(temp) / 'bundle'
                staging.mkdir()
                for name in names:
                    shutil.copy2(source / (names[name] if isinstance(names, dict) else name), staging / name)
                write_json(staging / 'cache.json', {
                    'key': self.key, 'origin': origin,
                    'files': {name: checksum(staging / name) for name in names},
                })
                for path in staging.iterdir():
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
                if self.path.exists():
                    shutil.rmtree(self.path)
                staging.replace(self.path)
                descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
