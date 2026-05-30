"""MkDocs hooks for light-curve-python docs."""

import os
import pathlib

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
