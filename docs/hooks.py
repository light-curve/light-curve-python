"""MkDocs hooks for light-curve-python docs."""

import copy
import functools
import os
import pathlib
import re
import xml.etree.ElementTree as ET

import material

_REPO = "light-curve/light-curve-python"
# GITHUB_HEAD_REF = PR source branch; GITHUB_REF_NAME = push branch/tag; fallback = main
_BRANCH = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME", "main")

_ICONS_DIR = pathlib.Path(material.__path__[0]) / "templates/.icons"


def _load_icon(bundle: str, name: str) -> str:
    svg_path = _ICONS_DIR / bundle / f"{name}.svg"
    return svg_path.read_text() if svg_path.exists() else ""


_ICON_DOWNLOAD = _load_icon("material", "download")
_ICON_COLAB = _load_icon("simple", "googlecolab")


def on_page_content(html, page, config, files, **kwargs):
    """Inject download and Google Colab buttons at the top of every notebook page."""
    if not page.file.src_path.endswith(".ipynb"):
        return html

    nb_path = f"docs/{page.file.src_path}"
    download_url = f"https://raw.githubusercontent.com/{_REPO}/{_BRANCH}/{nb_path}"
    colab_url = (
        f"https://colab.research.google.com/github/{_REPO}/blob/{_BRANCH}/{nb_path}"
    )

    buttons = (
        f'<p class="lc-nb-buttons">'
        f'<a href="{download_url}" class="md-button md-button--primary" download>'
        f'<span class="lc-nb-icon">{_ICON_DOWNLOAD}</span>Download notebook</a> '
        f'<a href="{colab_url}" class="md-button md-button--primary" target="_blank" rel="noopener">'
        f'<span class="lc-nb-icon">{_ICON_COLAB}</span>Run in Google Colab</a>'
        f"</p>\n"
    )
    return buttons + html


# ── Lockups ──────────────────────────────────────────────────────────────
#
# The designer's lockup is Licu with two lines of Cascadia Mono beside it. The
# arrangement suits any short two-line phrase, so a page asks for one by its
# wording alone:
#
#     ```lockup
#     pip install
#     light-curve
#     ```
#
# Both background variants are inlined and the stylesheet shows the matching
# one, the same way the header logo works. They are used exactly as exported:
# each already carries the right purple for its background, and the lettering
# colour the designer chose to go with it.

_SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

_LOGO_DIR = pathlib.Path(__file__).parent / "assets" / "logo"
_LOCKUP_VARIANTS = ("light", "dark")

# Cascadia Mono is monospaced: every glyph advances the same fraction of the em.
_CASCADIA_ADVANCE_EM = 1200 / 2048

# A longer fence around a shorter one is how Markdown quotes a fence verbatim,
# which is how the contributing guide shows what a lockup looks like. Those
# regions are left alone, or documenting the feature would invoke it.
_QUOTED_FENCE = re.compile(
    r"^(?P<fence>````+)[^\n]*\n.*?^(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

# All of these match a fenced block in the page's Markdown, not the artwork.
_LOCKUP_FENCE = re.compile(
    r"^```lockup[ \t]*\n(?P<body>.*?)\n```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
# ```mark
# mark-wide           <- a vendored asset, minus its -light/-dark suffix
# light-curve         <- optional, becomes the accessible label
# ```
_MARK_FENCE = re.compile(
    r"^```mark[ \t]*\n(?P<stem>[\w-]+)[ \t]*\n(?:(?P<label>[^\n]*)\n)?```[ \t]*$",
    re.MULTILINE,
)
_FONT_SIZE_RULE = re.compile(r"font-size:\s*(?P<px>[\d.]+)px")


def _svg_tag(name: str) -> str:
    return f"{{{_SVG_NS}}}{name}"


class LockupError(Exception):
    """The lockup artwork is not shaped the way the hook expects."""


@functools.cache
def _artwork(stem: str) -> ET.Element:
    """Parse one asset once; callers work on copies."""
    path = _LOGO_DIR / f"{stem}.svg"
    if not path.exists():
        raise LockupError(f"missing {path}")
    return ET.parse(path).getroot()


def _lockup(variant: str) -> ET.Element:
    """A lockup, checked for the two lines of lettering the hook fills in."""
    root = _artwork(f"lockup-{variant}")
    if len(_lettering(root)) != 2:
        raise LockupError(f"lockup-{variant}.svg: expected two lines of lettering")
    return root


def _origin_x(text: ET.Element) -> float:
    """Where a line starts, from the translate on its text element."""
    transform = text.get("transform", "")
    if not transform.startswith("translate("):
        raise LockupError(f"unexpected transform on a text element: {transform!r}")
    x, _, _ = (
        transform.removeprefix("translate(").removesuffix(")").strip().partition(" ")
    )
    return float(x)


def _lettering(root: ET.Element) -> list[tuple[ET.Element, float]]:
    """Each line as the tspan holding it, paired with the x it starts at.

    The lockup's two lines are the only text in the artwork, so their order in
    the document identifies them. Nothing else is needed to find them, which
    matters because Illustrator regenerates its class and id names on every
    export.
    """
    lines = []
    for text in root.iter(_svg_tag("text")):
        tspans = list(text.iter(_svg_tag("tspan")))
        if len(tspans) != 1:
            raise LockupError(
                f"a text element holds {len(tspans)} tspans, expected one"
            )
        lines.append((tspans[0], _origin_x(text)))
    return lines


def _advance(root: ET.Element) -> float:
    """User units per character, from the size the artwork sets its type in."""
    style = root.find(f".//{_svg_tag('style')}")
    match = _FONT_SIZE_RULE.search(style.text or "") if style is not None else None
    if match is None:
        raise LockupError("no font-size in the artwork's stylesheet")
    return float(match["px"]) * _CASCADIA_ADVANCE_EM


def _set_wording(root: ET.Element, lines: list[str]) -> None:
    """Replace the lettering, widening the canvas so the new wording fits.

    Everything else is left where it was drawn. The only thing that cannot
    stay as exported is the width: longer wording would run past the viewBox
    and simply not render. The type is monospaced at a known size, so the
    width it needs follows from its character count, and the right margin the
    designer left is preserved.

    The margin has to be read before the lettering is replaced, which is why
    both happen here rather than in the caller.
    """
    view_box = [float(value) for value in root.get("viewBox", "").split()]
    if len(view_box) != 4:
        raise LockupError(f"unusable viewBox: {root.get('viewBox')!r}")
    min_x, min_y, width, height = view_box

    advance = _advance(root)
    lettering = _lettering(root)

    def right_edge(texts: list[str]) -> float:
        return max(
            origin + len(text) * advance for (_, origin), text in zip(lettering, texts)
        )

    margin = width - right_edge([tspan.text or "" for tspan, _ in lettering])
    if margin < 0:
        raise LockupError("the artwork's own lettering overflows its viewBox")

    for (tspan, _), line in zip(lettering, lines):
        tspan.text = line

    new_width = round(right_edge(lines) + margin, 4)
    root.set("viewBox", f"{min_x:g} {min_y:g} {new_width:g} {height:g}")
    root.set("width", f"{new_width:g}")


def _uniquify_ids(root: ET.Element, token: str) -> None:
    """Suffix every id, so two lockups on one page cannot collide."""
    renames = {}
    for element in root.iter():
        old = element.get("id")
        if old is not None:
            renames[old] = f"{old}-{token}"
            element.set("id", renames[old])
    if not renames:
        return

    def retarget(value: str) -> str:
        for old, new in renames.items():
            value = value.replace(f"url(#{old})", f"url(#{new})")
        return value

    for element in root.iter():
        for name, value in list(element.items()):
            element.set(name, retarget(value))

    style = root.find(f".//{_svg_tag('style')}")
    if style is not None and style.text:
        style.text = retarget(style.text)


def _wrap(rendered: list[str], ratio: str, css_class: str) -> str:
    """Put both variants in a wrapper sized from the artwork's own proportions.

    The width is stated rather than left to `width: auto`, which on a block box
    means fill the container -- a definite width, so neither aspect-ratio nor
    the artwork's proportions would get a say and preserveAspectRatio would
    letterbox the drawing inside a page-wide box.

    min() caps it at the text column: a long phrase would otherwise run past
    the measure on pages that also carry a navigation sidebar. The aspect-ratio
    is what keeps the drawing proportional when that cap bites, since the
    height is then no longer the height asked for.

    The height stays the single shared knob; only the ratio is per-instance,
    and it comes from the artwork rather than from a constant.
    """
    return (
        f'<div class="{css_class}" '
        f'style="width: min(100%, calc(var(--lc-mark-height) * {ratio})); '
        f'aspect-ratio: {ratio}">'
        f"{''.join(rendered)}</div>"
    )


def _ratio(root: ET.Element) -> str:
    width, height = root.get("width"), root.get("viewBox").split()[3]
    return f"{width} / {height}"


def _render_lockup(lines: list[str], token: str) -> str:
    """Both background variants of the lockup, set in the given wording."""
    rendered, ratio = [], None
    for variant in _LOCKUP_VARIANTS:
        root = copy.deepcopy(_lockup(variant))
        _set_wording(root, lines)
        _uniquify_ids(root, f"{token}-{variant}")
        root.set("class", f"lc-logo--{variant}")
        if variant != _LOCKUP_VARIANTS[0]:
            root.set("aria-hidden", "true")
        else:
            root.set("role", "img")
            root.set("aria-label", " ".join(lines))
            ratio = _ratio(root)
        rendered.append(ET.tostring(root, encoding="unicode"))
    return _wrap(rendered, ratio, "lc-lockup")


def _render_mark(stem: str, label: str, token: str) -> str:
    """Both background variants of a plain mark, inlined.

    Inlined rather than referenced with <img> so that it goes through the same
    rendering path as a lockup. An SVG behind <img> is rasterised into the
    image box and snapped to whole device pixels, while an inline one is
    painted as vectors -- enough to draw the same artwork at visibly different
    sizes on two pages that ask for the same height.
    """
    rendered, ratio = [], None
    for variant in _LOCKUP_VARIANTS:
        root = copy.deepcopy(_artwork(f"{stem}-{variant}"))
        _uniquify_ids(root, f"{token}-{variant}")
        root.set("class", f"lc-logo--{variant}")
        if variant != _LOCKUP_VARIANTS[0]:
            root.set("aria-hidden", "true")
        else:
            root.set("role", "img")
            root.set("aria-label", label)
            ratio = _ratio(root)
        rendered.append(ET.tostring(root, encoding="unicode"))
    return _wrap(rendered, ratio, "lc-mark")


def on_page_markdown(markdown, page, config, files, **kwargs):
    """Expand the artwork fences into inline SVG."""
    if "```lockup" not in markdown and "```mark" not in markdown:
        return markdown

    counter = iter(range(1, 1000))

    def token() -> str:
        return f"{page.file.src_uri.replace('/', '-')}-{next(counter)}"

    def expand_lockup(match: re.Match) -> str:
        lines = match["body"].split("\n")
        if len(lines) != 2:
            raise LockupError(
                f"{page.file.src_path}: a lockup takes exactly two lines, got {len(lines)}"
            )
        return _render_lockup(lines, token())

    def expand_mark(match: re.Match) -> str:
        return _render_mark(match["stem"], match["label"] or "", token())

    def expand(chunk: str) -> str:
        chunk = _LOCKUP_FENCE.sub(expand_lockup, chunk)
        return _MARK_FENCE.sub(expand_mark, chunk)

    # Walk the document, expanding everything except the quoted fences.
    out, last = [], 0
    for quoted in _QUOTED_FENCE.finditer(markdown):
        out.append(expand(markdown[last : quoted.start()]))
        out.append(quoted.group())
        last = quoted.end()
    out.append(expand(markdown[last:]))
    return "".join(out)
