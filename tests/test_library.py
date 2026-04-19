from mcpzim.library import ZimKind, classify


def _c(**kw):
    base = {
        "filename": "x.zim",
        "name": "",
        "title": "",
        "creator": "",
        "publisher": "",
        "tags": [],
        "has_routing_entry": False,
        "has_streetzim_config": False,
    }
    base.update(kw)
    return classify(**base)


def test_wikipedia_by_name():
    assert _c(name="wikipedia_en_all", creator="Wikipedia") == ZimKind.WIKIPEDIA


def test_wikipedia_by_tag():
    assert _c(tags=["wikipedia", "_category:wikipedia"]) == ZimKind.WIKIPEDIA


def test_mdwiki_by_tag_medical():
    assert _c(name="mdwiki_en_all", tags=["medical", "mdwiki"]) == ZimKind.MDWIKI


def test_mdwiki_by_publisher():
    assert _c(publisher="WikiProjectMed", name="mdwiki_en_all") == ZimKind.MDWIKI


def test_streetzim_by_signature_entry():
    # Filename/name are neutral; the signature entry wins.
    assert _c(filename="foo.zim", has_routing_entry=True) == ZimKind.STREETZIM
    assert _c(filename="foo.zim", has_streetzim_config=True) == ZimKind.STREETZIM


def test_streetzim_beats_wikipedia_by_signature():
    # If somehow a streetzim ZIM is named like a Wikipedia ZIM, the signature
    # file still takes priority — we're matching on concrete contents.
    assert (
        _c(filename="wikipedia_en_all.zim", name="wikipedia_en_all", has_routing_entry=True)
        == ZimKind.STREETZIM
    )


def test_generic_fallback():
    assert _c(name="random_archive", creator="Someone") == ZimKind.GENERIC
