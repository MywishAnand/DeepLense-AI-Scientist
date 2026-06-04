from dlens.core import ping


def test_ping():
    assert ping() == "dlens ok"
