"""Shared cache publication is atomic and rejects damaged bundles."""

from concurrent.futures import ThreadPoolExecutor


from validation.shared_cache import SharedCache


def test_atomic_concurrent_publication_and_corruption(current_context, tmp_path):
    cache = SharedCache(current_context, "embeddings", "test-key")
    sources = []
    for i in range(6):
        source = tmp_path / str(i)
        source.mkdir()
        (source / "a").write_text(str(i))
        (source / "b").write_text(str(i))
        sources.append(source)
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(
            executor.map(lambda p: cache.publish(p, ["a", "b"], origin={"run_id": p.name}), sources)
        )
    restored = tmp_path / "restored"
    assert cache.restore(restored)
    assert (restored / "a").read_bytes() == (restored / "b").read_bytes()
    (cache.path / "a").write_text("corrupt")
    assert cache.restore(restored) is None
    cache.publish(sources[0], ["a", "b"], origin={"run_id": "0"})
    assert cache.valid()
    (cache.path / "cache.json").unlink()
    assert cache.restore(restored) is None
