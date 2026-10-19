class ExtractionStepError(RuntimeError):
    pass


class UnsupportedArtifactError(ExtractionStepError):
    """Existing artifacts outside the current contract must never be rewritten implicitly."""
