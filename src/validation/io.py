"""Durable validation artifact writes."""

from artifact_io import atomic_write_json as _write_json, atomic_write_jsonl as _write_jsonl


def atomic_write_json(path, value):
    _write_json(path, value, durable=True)


def atomic_write_jsonl(path, rows):
    _write_jsonl(path, rows, durable=True)
