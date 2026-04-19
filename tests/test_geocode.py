from mcpzim.geocode import normalize_prefix


def test_prefix_lowercases_and_truncates():
    assert normalize_prefix("New York") == "ne"


def test_prefix_pads_short_queries():
    assert normalize_prefix("a") == "a_"
    assert normalize_prefix("") == "__"


def test_prefix_replaces_non_alphanumeric():
    assert normalize_prefix("é1-foo") == "_1"
    assert normalize_prefix(" x") == "_x"


def test_prefix_keeps_digits_and_underscores():
    assert normalize_prefix("42nd Street") == "42"
    assert normalize_prefix("_private") == "_p"
