from pathlib import Path

from markus_exam_matcher.core.char_types import CharType
from markus_exam_matcher.image_processing import read_chars


def test_read_char_example():
    """Test read_chars on example student_info_num.jpg."""
    input_file = Path(__file__).parent / "file_fixtures" / "student_info_num.jpg"
    actual = read_chars.run(input_file, char_type=CharType.DIGIT)
    expected = "0001250981"
    assert actual == expected
