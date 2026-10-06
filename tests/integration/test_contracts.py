import pytest

from pipeline_runtime import ConfigError, RunContext


@pytest.mark.parametrize("name", ["..", "a/b", "resized_keyframes", "source_assets", "a\\b", ""])
def test_run_id_cannot_escape_or_claim_the_shared_frame_directory(current_context, name):
    with pytest.raises(ConfigError):
        RunContext.load(name, root=current_context.root)
