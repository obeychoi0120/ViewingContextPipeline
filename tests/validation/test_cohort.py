import pytest
from validation.cohort import load_metadata_titles
from validation.cohort_selection import CohortError

def test_title_csv_accepts_bom_and_splits_only_the_first_comma(tmp_path) -> None:
    path = tmp_path / "titles.csv"
    path.write_text("\ufeff1,A title, with commas\n2,Second title\n", encoding="utf-8")

    assert load_metadata_titles(path) == {
        "1": "A title, with commas",
        "2": "Second title",
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1,First\n01,Duplicate\n", "duplicate metadata title"),
        ("1,   \n01,Duplicate\n", "duplicate metadata title"),
        ("1,First\n01,   \n", "duplicate metadata title"),
        ("1,\n01,\n", "duplicate metadata title"),
        ("not-an-id,Title\n", "invalid metadata title row"),
        ("not-an-id,   \n", "invalid metadata title row"),
        ("1 Title\n", "missing comma"),
    ],
)
def test_title_csv_rejects_invalid_rows(tmp_path, text: str, message: str) -> None:
    path = tmp_path / "titles.csv"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(CohortError, match=message):
        load_metadata_titles(path)


def test_title_csv_omits_blank_titles_for_catalog_coverage_check(tmp_path) -> None:
    path = tmp_path / "titles.csv"
    path.write_text("\ufeff1,A title, with commas\n2,\n1849, \t\n", encoding="utf-8")

    assert load_metadata_titles(path) == {"1": "A title, with commas"}
