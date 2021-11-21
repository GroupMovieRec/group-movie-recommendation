from app import index


def test_index(app, client):
    assert index() == "Test 1"
