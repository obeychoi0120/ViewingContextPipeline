
import pytest

from extraction.semantic_graph import parse_or_repair_graph


TEXT = """[Entities]
person-1: person; long-haired; black, white and red shirt
cup1: cup; blue-green
[Relations]
person-1 -> standing-next-to -> cup1
[Context]
A blue-green, softly lit room.
[End]"""


@pytest.mark.parametrize(
    "text",
    [
        TEXT.replace("[Relations]", "[Entities]"),
        TEXT.replace("person-1: person", "person-1:"),
        TEXT.replace(" -> standing-next-to -> ", " - standing - next to - "),
    ],
)
def test_does_not_invent_discard_or_select_ambiguous_content(text):
    result = parse_or_repair_graph(text)
    assert result.graph is None and result.error
