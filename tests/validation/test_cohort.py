from validation.cohort import load_metadata_titles


def test_title_csv_omits_blank_titles_for_catalog_coverage_check(tmp_path) -> None:
    path = tmp_path / "titles.csv"
    path.write_text("\ufeff1,A title, with commas\n2,\n1849, \t\n", encoding="utf-8")

    assert load_metadata_titles(path) == {"1": "A title, with commas"}
