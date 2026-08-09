from nursery.hashing import stable_hash


def test_same_payload_gives_same_hash():
    assert stable_hash({"a": 1, "b": [2, 3]}) == stable_hash({"a": 1, "b": [2, 3]})


def test_key_order_does_not_change_hash():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_different_payload_gives_different_hash():
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_list_order_does_change_hash():
    assert stable_hash([1, 2]) != stable_hash([2, 1])


def test_hash_is_short_hex():
    h = stable_hash({"a": 1})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
