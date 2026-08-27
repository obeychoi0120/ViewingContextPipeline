import numpy as np

from validation.steps import _representations_match_catalog
from viewing_context_pipeline.runtime import write_json


def test_representation_cache_must_match_full_catalog(tmp_path) -> None:
    catalog = [{"item_id": "1"}, {"item_id": "2"}]
    item_index_path = tmp_path / "item_index.json"
    outputs = [tmp_path / f"{branch}.npz" for branch in ("graph_qwen", "graph_gemini", "desc")]
    write_json(item_index_path, {"1": 0, "2": 1})
    for path in outputs:
        np.savez_compressed(path, values=np.ones((2, 1024), dtype=np.float32))

    assert _representations_match_catalog(item_index_path, outputs, catalog, 1024)

    write_json(item_index_path, {"1": 0})
    assert not _representations_match_catalog(item_index_path, outputs, catalog, 1024)
