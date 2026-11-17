import zipfile

import numpy as np
import pytest

from src.load import COLUMNS, layer_from_name, layers_from_names, load_part, sniff
from src.synthetic import write_part


def test_there_are_40_unique_columns():
    assert len(COLUMNS) == 40
    assert len(set(COLUMNS)) == 40


def test_layer_number_from_file_name():
    assert layer_from_name("Layer0125.csv") == 125
    assert layer_from_name("part1/layer_7.csv.gz") == 7
    assert layer_from_name("data.csv") is None


def test_layer_numbers_use_the_number_that_changes():
    assert layers_from_names(["Part1_Layer002.csv", "Part1_Layer010.csv"]) == [2, 10]
    assert layers_from_names(["Layer2_Part1.csv", "Layer10_Part1.csv"]) == [2, 10]
    # no numbers at all: natural order
    assert layers_from_names(["b.csv", "a.csv"]) == [2, 1]


def test_sniff_header_and_separator():
    assert sniff("1,2.5,3e4") == (",", False)
    assert sniff("part,time,x") == (",", True)
    assert sniff("1 2.5 3") == (None, False)
    assert sniff("1\t2\t3") == ("\t", False)


def test_load_part_gives_back_what_was_written(dataset, data_dir):
    points = load_part(data_dir / "part1", layer_thickness_mm=0.1)
    original = dataset[1]
    assert len(points) == len(original)
    assert list(points["layer"].unique()) == list(range(1, 9))
    assert set(COLUMNS) <= set(points.columns)
    np.testing.assert_allclose(points["mp_area_100"], original["mp_area_100"], atol=1e-5)
    np.testing.assert_allclose(points["z"], points["layer"] * 0.1, rtol=1e-6)
    # sorted by layer, then time
    assert (np.diff(points["layer"]) >= 0).all()


def test_header_line_is_skipped(dataset, tmp_path):
    write_part(dataset[1], tmp_path / "with_header", header=True, gz=False)
    write_part(dataset[1], tmp_path / "no_header", header=False, gz=False)
    a = load_part(tmp_path / "with_header")
    b = load_part(tmp_path / "no_header")
    assert len(a) == len(b)
    np.testing.assert_allclose(a["x"], b["x"])


def test_load_from_zip_and_layer_range(dataset, tmp_path):
    write_part(dataset[2], tmp_path / "p", gz=False)
    zip_path = tmp_path / "part02.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in sorted((tmp_path / "p").iterdir()):
            zf.write(file, f"part02/{file.name}")
    points = load_part(zip_path, layers=(3, 5))
    assert sorted(points["layer"].unique()) == [3, 4, 5]
    assert (points["part"] == 2).all()


def test_empty_columns_are_kept_and_trailing_separator_is_dropped(tmp_path):
    # Like the real NIST layer 1: the last 21 columns (layer camera, XCT) are NaN.
    row = ",".join(["1", "77500"] + ["0.5"] * 17 + ["NaN"] * 21)
    folder = tmp_path / "nan_cols"
    folder.mkdir()
    (folder / "L0001.csv").write_text(row + "\n" + row + "\n")
    (folder / "L0002.csv").write_text(row + ",\n" + row + ",\n")  # trailing separator
    points = load_part(folder)
    assert len(points) == 4
    assert points["xct_m3"].isna().all()
    assert points["mp_area_100"].notna().all()


def test_wrong_column_count_gives_a_clear_error(tmp_path):
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "layer_1.csv").write_text("1,2,3\n4,5,6\n")
    with pytest.raises(ValueError, match="Expected 40 columns"):
        load_part(folder)
