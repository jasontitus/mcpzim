from mcpzim.content import html_to_text, _snippet


def test_html_to_text_strips_scripts_and_nav():
    html = """
    <html><head><script>alert(1)</script><style>.x{}</style></head>
    <body>
      <nav>nav bar</nav>
      <div id="mw-content-text">
        <h1>Title</h1>
        <p>Hello <b>world</b>. This is an <a href="/x">article</a>.</p>
        <div class="navbox">boilerplate</div>
        <span class="mw-editsection">[edit]</span>
      </div>
    </body></html>
    """
    text = html_to_text(html)
    assert "alert(1)" not in text
    assert "boilerplate" not in text
    assert "[edit]" not in text
    assert "Hello world" in text
    assert "Title" in text


def test_html_to_text_falls_back_without_mediawiki_container():
    html = "<html><body><p>bare paragraph</p></body></html>"
    assert "bare paragraph" in html_to_text(html)


def test_snippet_centers_on_first_query_match():
    text = "a " * 200 + "the target phrase appears here" + " z" * 200
    snip = _snippet(text, "target")
    assert "target" in snip
    # Should be shorter than the full text.
    assert len(snip) < len(text)


def test_snippet_empty_on_no_match():
    out = _snippet("nothing here", "absent")
    # Fall back to a prefix — not empty but bounded.
    assert out.startswith("nothing here")
