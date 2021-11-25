from src.app import index


def test_index():
    assert index() == "Test 1"
