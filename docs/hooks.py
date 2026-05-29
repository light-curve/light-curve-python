"""MkDocs hooks for light-curve-python docs."""

import pathlib

_REPO = "light-curve/light-curve-python"
_BRANCH = "main"

_ICONS_DIR = pathlib.Path(__file__).parent.parent / ".venv/lib"


def _load_icon(bundle: str, name: str) -> str:
    """Return inline SVG for a Material/Simple icon, stripped of the XML declaration."""
    # Walk .venv/lib to find the icons directory regardless of Python version
    for icons_dir in _ICONS_DIR.rglob("material/templates/.icons"):
        svg_path = icons_dir / bundle / f"{name}.svg"
        if svg_path.exists():
            return svg_path.read_text()
    return ""


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
        f'<a href="{download_url}" class="md-button" download>'
        f'<span class="lc-nb-icon">{_ICON_DOWNLOAD}</span>Download notebook</a> '
        f'<a href="{colab_url}" class="md-button md-button--primary" target="_blank" rel="noopener">'
        f'<span class="lc-nb-icon">{_ICON_COLAB}</span>Run in Google Colab</a>'
        f"</p>\n"
    )
    return buttons + html
