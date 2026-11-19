"""The browser's markdown renderer (static/markdown.js), run under Node.

Skipped where Node isn't installed; GitHub's runners have it.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import MakeClient

NODE = shutil.which("node")
HARNESS = Path(__file__).parent / "markdown_harness.js"

needs_node = pytest.mark.skipif(NODE is None, reason="needs node")


def render(markdown: str) -> str:
    assert NODE is not None
    result = subprocess.run(  # noqa: S603 - fixed command, input on stdin
        [NODE, str(HARNESS)], input=markdown, capture_output=True, text=True, check=True
    )
    return result.stdout


@needs_node
def test_paragraphs_and_emphasis() -> None:
    html = render("Use the **laser** with *care* and ~~not~~ `code`.\nSame paragraph.\n\nNext")
    assert html == (
        "<p>Use the <strong>laser</strong> with <em>care</em> and <del>not</del> "
        "<code>code</code>. Same paragraph.</p><p>Next</p>"
    )


@needs_node
def test_raw_html_is_text() -> None:
    html = render('<script>alert(1)</script> <img src=x onerror="alert(1)">')
    assert "<script" not in html
    assert "<img" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


@needs_node
def test_links_allow_only_http() -> None:
    html = render(
        "[wiki](https://wiki.example.org/A_(b)) [bad](javascript:alert(1)) https://x.org/y."
    )
    new_tab = 'target="_blank" rel="noopener noreferrer"'
    assert f'<a href="https://wiki.example.org/A_(b)" {new_tab}>wiki</a>' in html
    assert "javascript" not in html
    assert ">bad<" not in html
    assert " bad " in html
    assert f'<a href="https://x.org/y" {new_tab}>https://x.org/y</a>.' in html


@needs_node
def test_citations_link_to_known_sources() -> None:
    html = render("Needs training [1]. See also [2, 3].")
    assert '<sup class="cite">[<a href="#source-1">1</a>]</sup>' in html
    assert '<sup class="cite">[<a href="#source-2">2</a>, 3]</sup>' in html


@needs_node
def test_headings_lists_and_nesting() -> None:
    html = render("## Steps\n\n1. Book\n2. Attend\n   - glasses\n   - shoes\n3. Done\n\n- a\n- b")
    assert html == (
        '<h4 class="md-heading">Steps</h4>'
        "<ol><li>Book</li><li>Attend<ul><li>glasses</li><li>shoes</li></ul></li><li>Done</li></ol>"
        "<ul><li>a</li><li>b</li></ul>"
    )


@needs_node
def test_ordered_list_start_and_loose_items() -> None:
    html = render("3. three\n\n4. four")
    assert html == '<ol start="3"><li><p>three</p></li><li><p>four</p></li></ol>'


@needs_node
def test_fenced_code_is_literal() -> None:
    html = render('```bash\necho "<hi>" *not em* [1]\n```')
    assert html == '<pre class="md-code"><code>echo "&lt;hi&gt;" *not em* [1]</code></pre>'


@needs_node
def test_table_quote_and_rule() -> None:
    html = render("> **Note**\n\n| Tool | OK |\n|---|:-:|\n| Drill \\| press | yes |\n\n---")
    assert html == (
        "<blockquote><p><strong>Note</strong></p></blockquote>"
        '<div class="md-table"><table><thead><tr><th>Tool</th><th align="center">OK</th></tr>'
        '</thead><tbody><tr><td>Drill | press</td><td align="center">yes</td></tr></tbody>'
        "</table></div><hr>"
    )


@needs_node
def test_snake_case_is_not_emphasis() -> None:
    assert render("set snake_case_name now") == "<p>set snake_case_name now</p>"


def test_ui_loads_the_renderer_before_the_app(make_client: MakeClient) -> None:
    with make_client() as client:
        page = client.get("/").text
        assert client.get("/static/markdown.js").status_code == 200
    assert page.index("/static/markdown.js") < page.index("/static/app.js")
